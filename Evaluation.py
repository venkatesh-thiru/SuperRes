import os
import torchio as tio
from torchio import AFFINE,DATA
import numpy as np
import random
import torch
from MultiScaleExperiment import MultiScale
import pandas as pd
import matplotlib.pyplot as plt
from torchio.transforms import Compose,ZNormalization,RescaleIntensity,RandomNoise
from sewar.full_ref import uqi,ssim
import torch.nn.functional as F
from scipy.ndimage import zoom
from tqdm import tqdm
from RRDB import RRDB
import PIL
from IPython.display import display
import json
from skimage.metrics import mean_squared_error,normalized_root_mse,peak_signal_noise_ratio,structural_similarity
from sewar.full_ref import uqi
import pytorch_ssim
import os
from torchio.transforms import Compose,ZNormalization,RescaleIntensity,RandomNoise
import pandas as pd
import torchio as tio
import numpy as np
import json
# from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from RRDB import RRDB
from tqdm import tqdm
from SPSR_GG import SPSR_GG



# HELPERS

def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(1, len(slices))
    fig.set_size_inches(15, 7.5)
    for i in range(len(slices)):
        axes[i].imshow(slices[i].T, cmap="gray", origin="lower")
    plt.show()


def plot_images(inp_np):
    slice_0 = inp_np[int(inp_np.shape[0] / 2), :, :]
    slice_1 = inp_np[:, int(inp_np.shape[1] / 2), :]
    slice_2 = inp_np[:, :, int(inp_np.shape[2] / 2)]
    show_slices([slice_0, slice_1, slice_2])


def get_middle_patch(inp_np, axis=0):
    if axis == 0:
        return inp_np[int(inp_np.shape[0] / 2), :, :]
    elif axis == 1:
        return inp_np[:, int(inp_np.shape[1] / 2), :]
    elif axis == 2:
        return inp_np[:, :, int(inp_np.shape[2] / 2)]


def super_sample(model, patch_loader, aggregator):
    model.eval()
    with torch.no_grad():
        for batch in (patch_loader):
            inputs = batch['interpolated']['data'].cuda()
            logits = model(inputs)
            location = batch[tio.LOCATION]
            aggregator.add_batch(logits, location)
    return aggregator


def load_model_weights(kind, weight_path):
    if kind == 'RRDB':
        model = RRDB(nChannels=1, nDenseLayers=6, nInitFeat=6, GrowthRate=12, featureFusion=True,
                     kernel_config=[3, 3, 3, 3]).cuda()
    elif kind == 'MSF':
        model = MultiScale(nChannels=1, nDenseLayers=6, nInitFeat=6, GrowthRate=12).cuda()

    elif kind == 'SPSR':
        model = SPSR_GG(nChannels=1,nDenseLayers=6,nInitFeat=6,GrowthRate=12,kernel_config=[3,3,3,3]).cuda()
    weights = torch.load(weight_path)
    model.load_state_dict(weights['model_state_dict'])

    return model


def patcher(image_path, fold):
    DATA_DIR = 'DATA/IXI-T1'
    ground_truth_path = os.path.join(DATA_DIR, "Actual_Images")
    interpolated_path = os.path.join(DATA_DIR, f"Interpolated/{fold}")
    test_transform = Compose([RescaleIntensity((0, 1))])

    subject = tio.Subject(
        ground_truth=tio.ScalarImage(os.path.join(ground_truth_path, image_path)),
        interpolated=tio.ScalarImage(os.path.join(interpolated_path, image_path))
    )
    test_ds = tio.SubjectsDataset([subject], transform=test_transform)

    sample = test_ds[0]

    patch_size = 64, 64, 64
    patch_overlap = 10, 10, 10

    grid_sampler = tio.inference.GridSampler(sample, patch_size, patch_overlap)
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
    aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
    return sample, grid_sampler, patch_loader, aggregator


def get_metric(gt, interp):
    metric_dict = {
        'MSE': mean_squared_error(gt, interp),
        'NRMSE': normalized_root_mse(gt, interp, normalization='min-max'),
        'PSNR': peak_signal_noise_ratio(gt, interp, data_range=gt.max() - gt.min()),
        'SSIM': structural_similarity(gt, interp, data_range=gt.max() - gt.min()),
        'SSIM5': structural_similarity(gt, interp, data_range=gt.max() - gt.min(),win_size=5),
        'SSIM7': structural_similarity(gt, interp, data_range=gt.max() - gt.min(),win_size=7),
        'SSIM11': structural_similarity(gt, interp, data_range=gt.max() - gt.min(),win_size=11),
        'UQI': uqi(gt, interp, ws=11),
        'SD of difference': np.std(gt - interp),
    }

    return metric_dict


def zoom_image(comp, target_shape, base_type=0):
    #     comp = to_interp_d[0].compressed.data.squeeze().numpy()
    #     gt   = to_interp_d[0].gt.data.squeeze().numpy()
    zoom_factor = target_shape / np.array(comp.shape)
    zoomed = zoom(comp, zoom_factor, order=base_type)

    return zoomed


def read_json_extract_value(path, sample):
    with open(path, 'r') as outfile:
        metrics = json.load(outfile)
    metrics = metrics.replace("'", "\"")
    metrics = json.loads(str(metrics))
    return metrics['results'][sample]


def metrics_string(method, metrics):
    return f"\nSSIM : {round(metrics[method]['SSIM'], 4)}|PSNR : {round(metrics[method]['PSNR'], 4)}\nMSE : {round(metrics[method]['MSE'], 4)}|NRMSE : {round(metrics[method]['NRMSE'], 4)} "


def make_plot(stacks, sample, fold, gt_spacing, comp_spacing):
    metrics = {
        'NN': read_json_extract_value(f'results/baseline_results/{fold}/NN_interpolation_metrics.json', sample),
        'SINC': read_json_extract_value(f'results/baseline_results/{fold}/sinc_interpolation_metrics.json', sample),
        'BILINEAR': read_json_extract_value(f'results/baseline_results/{fold}/Bil_interpolation_metrics.json', sample),
        'BICUBIC': read_json_extract_value(f'results/baseline_results/{fold}/BiC_interpolation_metrics.json', sample),
        'RRDB': read_json_extract_value(f'results/RRDB/RRDB_{fold}_SSIM_metrics.json', sample),
        'MSF': read_json_extract_value(f'results/MSF/MSF_{fold}_SSIM_metrics.json', sample)
    }
    nrow = 3
    ncol = 4

    fig, ax = plt.subplots(3, 4, figsize=(20, 15), dpi=80, sharex=True, sharey=True)

    for row in range(0, nrow):
        for col in range(0, ncol):
            ax[row][col].set_axis_off()

    fig.suptitle(f"{sample}\n{fold} downsampling")

    ax[0][0].imshow(stacks['ground_truth'].T, interpolation='nearest', origin='lower', cmap='gray')
    ax[0][0].set_title(f"Original \n {round(gt_spacing[0], 2)}X{round(gt_spacing[1], 2)}X{round(gt_spacing[2], 2)}")

    ax[0][1].imshow(stacks['compressed'].T, interpolation='nearest', origin='lower', cmap='gray')
    ax[0][1].set_title(
        f"Compressed \n {round(comp_spacing[0], 2)}X{round(comp_spacing[1], 2)}X{round(comp_spacing[2], 2)}")

    ax[1][0].imshow(stacks['sinc'].T, interpolation='nearest', origin='lower', cmap='gray')
    ax[1][0].set_title(r"$\bfSINC$" + metrics_string('SINC', metrics))

    ax[1][1].imshow(stacks['nearest neighbor'].T, interpolation='nearest', origin='lower', cmap='gray')
    ax[1][1].set_title(r"$\bfNN$" + metrics_string('NN', metrics))

    ax[1][2].imshow(stacks['bilinear'].T, interpolation='nearest', origin='lower', cmap='gray')
    ax[1][2].set_title(r"$\bfbilinear$" + metrics_string('BILINEAR', metrics))

    ax[1][3].imshow(stacks['bicubic'].T, interpolation='nearest', origin='lower', cmap='gray')
    ax[1][3].set_title(r"$\bfbicubic$" + metrics_string('BICUBIC', metrics))

    ax[2][0].imshow(stacks['RRDB'].T, interpolation='nearest', origin='lower', cmap='gray')
    ax[2][0].set_title(r"$\bfRRDB$" + metrics_string('RRDB', metrics))

    ax[2][1].imshow(stacks['MSF'].T, interpolation='nearest', origin='lower', cmap='gray')
    ax[2][1].set_title(r"$\bfMSF$" + metrics_string('MSF', metrics))

    plt.show()


if __name__ == "__main__":
    # code
    meta_df_path = "Train_Test_Val_split_IXI-T1.csv"
    meta_df = pd.read_csv(meta_df_path)
    images = meta_df[meta_df.Type == 'Test'].file_names.values


    losses = ['SSIM']
    folds = ['2fold','2d5fold','3fold','3d5fold','4fold']

    for loss in losses:
        for fold in folds:
            RRDB_metrics = {'model':'RRDB','Loss_function':loss}
            result = {}
            for sample in tqdm(images,position = 0):
                subject,_,patch_loader,aggregator = patcher(sample,fold)
                model = load_model_weights(kind = 'RRDB',weight_path = f'Models/singlularModel/Discrete Scaling/SSARAVAN_Densenet_with_fusion_crossResolution_{loss}.pth')
                aggregator = super_sample(model,patch_loader,aggregator)
                final_image = aggregator.get_output_tensor()
    #             plot_images(final_image.squeeze().numpy())
                result[sample] = get_metric(subject['ground_truth']['data'].squeeze().numpy(),final_image.squeeze().numpy())

            RRDB_metrics['results'] = result
            data = json.dumps(str(RRDB_metrics))
            data_j = json.loads(data)
            with open(f'results/singularModel/Discrete Scaling/{fold}/RRDB_crossResolution_{loss}_metrics.json','w') as outfile:
                json.dump(data_j,outfile)
            print('============================================================================================================')
            print(f'{loss}================{fold}====================DONE!!!!!!!!!!')

