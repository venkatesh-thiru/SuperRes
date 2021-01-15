import matplotlib.pyplot as plt
import os
from pathlib import Path
import torchio as tio
from sklearn.model_selection import train_test_split



def show_slices(slices):
    """ Function to display row of image slices """
    fig, axes = plt.subplots(len(slices), len(slices[0]))
    fig.set_size_inches(15,7.5)
    for row in range(len(slices)):
        for column in range(len(slices[0])):
            axes[row][column].imshow(slices[row][column].T, cmap="gray", origin="lower")


def plot_images(inp_np,res_np):
    slice_0 = inp_np[int(inp_np.shape[0]/2), :, :]
    slice_1 = inp_np[:, int(inp_np.shape[1]/2), :]
    slice_2 = inp_np[:, :, int(inp_np.shape[2]/2)]
    nslice_0 = res_np[int(res_np.shape[0]/2), :, :]
    nslice_1 = res_np[:, int(res_np.shape[1]/2), :]
    nslice_2 = res_np[:, :, int(res_np.shape[2]/2)]
    show_slices([[slice_0, slice_1, slice_2],
                 [nslice_0, nslice_1, nslice_2]])


def train_test_val_split():
    ground_truths = Path("/nfs1/ssaravan/Data/Actual_Images")
    ground_paths = sorted(ground_truths.glob('*.nii.gz'))
    compressed_dirs = Path("/nfs1/ssaravan/Data/Interpolated")
    compressed_paths = sorted(compressed_dirs.glob('*.nii.gz'))
    subjects = []
    for gt,comp in zip(ground_paths,compressed_paths):
        subject = tio.Subject(
                ground_truth = tio.ScalarImage(gt),
                compressed = tio.ScalarImage(comp),
                )
        subjects.append(subject)
    train_split,test_split = train_test_split(subjects,test_size=0.3)
    test_split,validation_split = train_test_split(test_split,test_size=0.2)
    return train_split,test_split,validation_split
