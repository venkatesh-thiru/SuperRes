import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
import os
from pathlib import Path
import torchio as tio
from sklearn.model_selection import train_test_split
import pandas as pd




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


def generate_subjects(file_list,intensity):
    ground_truths = os.path.join(intensity,"Actual_Images")
    interpolated = os.path.join(intensity,"Interpolated")
    subjects = []
    for file in file_list:
        gt_path = os.path.join(ground_truths,file)
        int_path = os.path.join(interpolated,file)
        subject = tio.Subject(
            ground_truth=tio.ScalarImage(gt_path),
            interpolated=tio.ScalarImage(int_path),
        )
        subjects.append(subject)
    return subjects

def train_test_val_split(csv_file,intensity):
    file_df = pd.read_csv(csv_file)
    train_files = file_df[file_df['Type'] == 'Train'].file_names.values
    test_files = file_df[file_df['Type'] == 'Test'].file_names.values
    val_files = file_df[file_df['Type'] == 'Validation'].file_names.values
    train_split = generate_subjects(train_files,intensity)
    test_split = generate_subjects(test_files, intensity)
    validation_split = generate_subjects(val_files, intensity)
    return train_split,test_split,validation_split

def generate_train_test_val_csv(intensity):
    files = os.listdir(os.path.join(intensity,"Actual_Images"))
    df = pd.DataFrame(files, columns=['file_names'])
    train_split, test_split = train_test_split(files, test_size=0.3, shuffle=False)
    test_split, validation_split = train_test_split(test_split, test_size=0.4, shuffle=False)

    df.loc[df['file_names'].isin(train_split), 'Type'] = "Train"
    df.loc[df['file_names'].isin(test_split), 'Type'] = "Test"
    df.loc[df['file_names'].isin(validation_split), 'Type'] = "Validation"

    print(len(train_split), len(test_split), len(validation_split))

    df.to_csv(f"Train_Test_Val_split_{intensity}.csv", index=False)

# if __name__ == "__main__":
#     generate_train_test_val("Train_Test_Val_split.csv","IXI-T1")
