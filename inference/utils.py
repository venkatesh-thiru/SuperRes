import os
import torchio as tio
from torchio import AFFINE,DATA
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torchio.transforms import Compose,ZNormalization,RescaleIntensity,RandomNoise
from sewar.full_ref import uqi,ssim
import torch.nn.functional as F
from scipy.ndimage import zoom
from tqdm import tqdm
import PIL
import json
from skimage.metrics import mean_squared_error,normalized_root_mse,peak_signal_noise_ratio,structural_similarity
from sewar.full_ref import uqi
import os
import pandas as pd
import torchio as tio
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchcomplex.nn.functional import interpolate
import random
from mpl_toolkits.axes_grid1 import make_axes_locatable


def show_slices(slices,title = None):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    fig.set_size_inches(15, 7.5)
    for i in range(len(slices)):
        axes[i].imshow(slices[i].T, cmap="gray", origin="lower")

    if title:
        plt.title(title)
    plt.show()


def plot_images(inp_np,title=None):
    slice_0 = inp_np[int(inp_np.shape[0] / 2), :, :]
    slice_1 = inp_np[:, int(inp_np.shape[1] / 2), :]
    slice_2 = inp_np[:, :, int(inp_np.shape[2] / 2)]
    show_slices([slice_0, slice_1, slice_2],title)


def get_middle_patch(inp_np, axis=2):
    if axis == 0:
        return inp_np[int(inp_np.shape[0] / 2), :, :]
    elif axis == 1:
        return inp_np[:, int(inp_np.shape[1] / 2), :]
    elif axis == 2:
        return inp_np[:, :, int(inp_np.shape[2] / 2)]


def super_sample(model, patch_loader, aggregator):
    model.eval()
    with torch.no_grad():
        for batch in patch_loader:
            inputs = batch['interpolated']['data'].cuda()
            logits = model(inputs)
            location = batch[tio.LOCATION]
            aggregator.add_batch(logits, location)
    return aggregator

def read_and_patch(datadir="/home/venky/LANShareDownloads/12cp clean code/DATA/IXI-T1", scan = "IXI504-Guys-1025-T1.nii.gz", fold="2d5fold", patch_size=64, overlap_sizes=10, batch_size = 7):

    HR_path = os.path.join(datadir, "Actual_Images", scan)
    LR_path = os.path.join(datadir, "Compressed", fold, scan)

    LR_image = tio.ScalarImage(LR_path)
    HR_image = tio.ScalarImage(HR_path)
    interp_image = interpolate(LR_image[DATA].unsqueeze(dim=0), HR_image[DATA].squeeze().shape).squeeze(dim = 0)

    test_transform = Compose([RescaleIntensity((0, 1))])
    subject = tio.Subject(
        ground_truth = HR_image,
        interpolated = tio.ScalarImage(tensor = interp_image, affine = HR_image[AFFINE]),
        # low_res = LR_image
    )
    subject_ds = tio.SubjectsDataset([subject], transform=test_transform)
    sample = subject_ds[0]

    patch_size = 64,64,64
    patch_overlap = 10,10,10
    batch_size = 7

    grid_sampler = tio.inference.GridSampler(sample, patch_size, patch_overlap)
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")

    return sample, grid_sampler, patch_loader, aggregator

def prepare_model(model):
    # Enables Dropout
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    return model

def compute_uncertainty(model, patch_loader, aggregator, sample_size = 10):
    samples = []

    for _ in tqdm(range(sample_size)):
        # Sets the seed for each patching and aggregation
        seed = random.sample(range(1, 100), 1)[0]
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        uncertainty_model = prepare_model(model)
        samples.append(super_sample(uncertainty_model, patch_loader, aggregator).get_output_tensor())
    samples = torch.cat(samples, dim = 0)
    return torch.mean(samples, dim = 0), torch.var(samples, dim = 0)

def make_comparison_plot(subject, final_image, variance_map):
    slice_gt = get_middle_patch(subject['ground_truth']['data'].squeeze().detach().cpu().numpy())
    slice_interpolation = get_middle_patch(subject['interpolated']['data'].squeeze().detach().cpu().numpy())
    slice_predicted = get_middle_patch(final_image.squeeze().detach().cpu().numpy())

    slice_diff_gtvinterp = get_middle_patch(torch.abs(subject['interpolated']['data']-subject['ground_truth']['data']).squeeze().detach().cpu().numpy())
    slice_diff_gtvpred = get_middle_patch(torch.abs(subject['interpolated']['data']-final_image).squeeze().detach().cpu().numpy())
    slice_uncertainty = get_middle_patch(variance_map.squeeze().detach().cpu().numpy())


    fig, ax = plt.subplots(nrows = 1, ncols = 6, figsize = (30, 5))
    divider = make_axes_locatable(ax[5])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax[0].imshow(slice_gt.T,interpolation = 'nearest',origin = 'lower',cmap = 'gray')
    ax[0].set_axis_off()
    ax[0].set_title("HR")

    ax[1].imshow(slice_interpolation.T,interpolation = 'nearest',origin = 'lower',cmap = 'gray')
    ax[1].set_axis_off()
    ax[1].set_title("Interp")

    ax[2].imshow(slice_predicted.T,interpolation = 'nearest',origin = 'lower',cmap = 'gray')
    ax[2].set_axis_off()
    ax[2].set_title("SR")

    ax[3].imshow(slice_diff_gtvinterp.T,interpolation = 'nearest',origin = 'lower',cmap = 'gray', vmin = 0, vmax = 1)
    ax[3].set_axis_off()
    ax[3].set_title("|gt - interp|")

    ax[4].imshow(slice_diff_gtvpred.T,interpolation = 'nearest',origin = 'lower',cmap = 'gray', vmin = 0, vmax = 1)
    ax[4].set_axis_off()
    ax[4].set_title("|HR - SR|")

    im = ax[5].imshow(slice_uncertainty.T, interpolation = 'nearest',origin = 'lower',cmap = 'gray')
    ax[5].set_axis_off()
    ax[5].set_title("Uncertainty")

    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()