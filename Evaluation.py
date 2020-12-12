import torch
import torchio
import matplotlib.pyplot as plt
from utils import plot_images
import DenseNetModel
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import torchio as tio
from torchio.transforms import Compose,ZNormalization,RescaleIntensity
from tqdm import tqdm
from torchio import AFFINE,DATA
from skimage.metrics import structural_similarity as ssim_sklearn
import random
import pickle
import seaborn as sns
import pandas as pd

state_dict = torch.load("Models/DenseNet varying kernel with scale augmentation/denseNet3D_torchIO_patch_32_samples_20_ADAMOptim_50Epochs_BS12_GlorotWeights_SSIM_1511.pth")
model = DenseNetModel.DenseNet(num_init_features=4,growth_rate=6,block_config=(6,6,6)).to("cuda")
model.load_state_dict(state_dict["model_state_dict"])
ground_truths = Path("IXI-T1/Actual_Images")
ground_paths = sorted(ground_truths.glob('*.nii.gz'))
compressed_dirs = [sorted(Path((os.path.join("IXI-T1",comp))).glob('*.nii.gz')) for comp in os.listdir("IXI-T1") if "Compressed" in comp]
validation_batch_size = 12
test_transform = Compose([RescaleIntensity((0,1))])

def test_network(sample):
    patch_size = 48,48,48
    patch_overlap = 4,4,4
    model.eval()
    grid_sampler = tio.inference.GridSampler(sample,patch_size,patch_overlap)
    patch_loader = torch.utils.data.DataLoader(grid_sampler,int(validation_batch_size/4))
    aggregator = tio.inference.GridAggregator(grid_sampler,overlap_mode="average")
    with torch.no_grad():
        for batch in patch_loader:
            inputs = batch["compressed"][DATA].to("cuda")
            logits = model(inputs)
            location = batch[tio.LOCATION]
            aggregator.add_batch(logits,location)
    model.train()
    result = aggregator.get_output_tensor()
    original, compressed = sample.ground_truth["data"].squeeze(), sample.compressed["data"].squeeze()
    result = torch.squeeze(result)
    original,compressed,result = original.detach().cpu().numpy(),compressed.detach().cpu().numpy(),result.detach().cpu().numpy()
    ssim_val = ssim_sklearn(original,result,data_range=original.max()-original.min())
    return original,compressed,result,ssim_val

def generate_test_subjects():
    global_list = []
    for compressed_paths in compressed_dirs:
        subjects = []
        for gt,comp in zip(ground_paths,compressed_paths):
            subject = tio.Subject(
                    ground_truth = tio.ScalarImage(gt),
                    compressed = tio.ScalarImage(comp),
                    )
            subjects.append(subject)
        train_split,test_split = train_test_split(subjects,test_size=0.3)
        test_split,validation_split = train_test_split(test_split,test_size=0.2)
        global_list.append(test_split)
    return global_list

def calculate_ssim(global_list):
    global_ssim = []
    for i,scale_factors in enumerate(global_list):
        set_ssim = []
        test_dataset = tio.SubjectsDataset(scale_factors,transform=test_transform)
        for sample in tqdm(test_dataset):
            _,_,_,ssim_val = test_network(sample)
            set_ssim.append(ssim_val)
        global_ssim.append(set_ssim)
    return global_ssim

def plot_data(dictionary):
    df = pd.DataFrame.from_dict(dictionary)
    sns.boxplot(data=df)
    plt.show()


def save_results(ssims):
    compressed_dirs = [comp for comp in os.listdir("IXI-T1") if "Compressed" in comp]
    dictionary = dict(zip(compressed_dirs, ssims))
    with open("ssims_constant_scale.data", "wb") as file_handle:
        pickle.dump(dictionary, file_handle)
    plot_data(dictionary)

def read_dict(path):
    with open(path, 'rb') as handle:
        dict = pickle.load(handle)
    return dict

if __name__ == "__main__":
    test_subjects = generate_test_subjects()
    ssims = calculate_ssim(test_subjects)
    save_results(ssims)

