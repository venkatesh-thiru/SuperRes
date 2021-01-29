import torch
import torchio
import matplotlib.pyplot as plt
from utils import train_test_val_split
import DenseNetModel
from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import torchio as tio
from torchio.transforms import Compose,RescaleIntensity
from tqdm import tqdm
from torchio import AFFINE,DATA
from skimage.metrics import structural_similarity as ssim_sklearn
import random
import pickle
import seaborn as sns
import pandas as pd

state_dict = torch.load("/nfs1/ssaravan/code/Models/Trial_DenseNet_run_T2_FixedKernel_3.pth")
model = DenseNetModel.DenseNet(num_init_features=4,growth_rate=6,block_config=(6,6,6)).cuda()
model.load_state_dict(state_dict["model_state_dict"])
test_transform = Compose([RescaleIntensity((0,1))])
validation_batch_size = 12

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

# def generate_test_subjects():
#     global_list = []
#     for compressed_paths in compressed_dirs:
#         subjects = []
#         for gt,comp in zip(ground_paths,compressed_paths):
#             subject = tio.Subject(
#                     ground_truth = tio.ScalarImage(gt),
#                     compressed = tio.ScalarImage(comp),
#                     )
#             subjects.append(subject)
#         train_split,test_split = train_test_split(subjects,test_size=0.3)
#         test_split,validation_split = train_test_split(test_split,test_size=0.2)
#         global_list.append(test_split)
#     return global_list
def calculate_ssim(test_subjects):
    ssim_list = []
    test_dataset = tio.SubjectsDataset(test_subjects,transform=test_transform)
    for sample in tqdm(test_dataset):
        _,_,_,ssim_val = test_network(sample)
        ssim_list.append(ssim_val)
        print(ssim_val)
    return ssim_list

def plot_data(dictionary):
    df = pd.DataFrame.from_dict(dictionary)
    sns.boxplot(data=df)
    plt.show()


def save_results(ssims):
    dictionary = {"SSIM" : ssims}
    with open("DenseNet_run_T2_FixedKernel_3.data", "wb") as file_handle:
        pickle.dump(dictionary, file_handle)
    plot_data(dictionary)

def read_dict(path):
    with open(path, 'rb') as handle:
        dict = pickle.load(handle)
    return dict

if __name__ == "__main__":
    _,test_subjects,_ = train_test_val_split("/nfs1/ssaravan/code/Train_Test_Val_split_IXI-T2.csv","IXI-T2")
    ssims = calculate_ssim(test_subjects)
    save_results(ssims)

