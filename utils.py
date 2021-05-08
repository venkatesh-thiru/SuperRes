import matplotlib.pyplot as plt
import numpy as np
import os
import torchio as tio
from sklearn.model_selection import train_test_split
import pandas as pd
import random




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


def generate_subjects(file_list,intensity,folders,path,repeats = 2):
    ground_truths = os.path.join(path,intensity,"Actual_Images")
    plist = [1/len(folders) for i in range(0,len(folders))]
    subjects = []
    seed_number = 42
    for i in range(0,repeats):
        np.random.seed(seed_number)
        interpolated_list = []
        fold_list = []
        for file in file_list:
            fold= np.random.choice(folders,p = plist)
            fold_list.append(fold)
            interpolated = os.path.join(path,intensity,"Interpolated",fold)
            interpolated_list.append(interpolated)

        for i,file in enumerate(file_list):
            gt_path = os.path.join(ground_truths,file)
            int_path = os.path.join(interpolated_list[i],file)
            subject = tio.Subject(
                ground_truth=tio.ScalarImage(gt_path),
                interpolated=tio.ScalarImage(int_path),
            )
            subjects.append(subject)
        seed_number += 1
    return subjects


def train_test_val_split(csv_file,path,intensity,folders = "2fold",repeats = 1):
    file_df = pd.read_csv(csv_file)
    train_files = file_df[file_df['Type'] == 'Train'].file_names.values
    test_files = file_df[file_df['Type'] == 'Test'].file_names.values
    val_files = file_df[file_df['Type'] == 'Validation'].file_names.values
    train_split = generate_subjects(train_files,intensity,folders,path,repeats = repeats)
    test_split = generate_subjects(test_files, intensity,folders,path,repeats = repeats)
    validation_split = generate_subjects(val_files, intensity,folders,path,repeats = repeats)
    return train_split,test_split,validation_split


def generate_train_test_val_csv(intensity):
    files = os.listdir(os.path.join("DATA",intensity,"Actual_Images"))
    df = pd.DataFrame(files, columns=['file_names'])
    train_split, test_split = train_test_split(files, test_size=0.3, shuffle=False)
    test_split, validation_split = train_test_split(test_split, test_size=0.4, shuffle=False)
    df.loc[df['file_names'].isin(train_split), 'Type'] = "Train"
    df.loc[df['file_names'].isin(test_split), 'Type'] = "Test"
    df.loc[df['file_names'].isin(validation_split), 'Type'] = "Validation"
    print(len(train_split), len(test_split), len(validation_split))
    df.to_csv(f"Train_Test_Val_split_{intensity}.csv", index=False)

if __name__ == "__main__":
    folders = ["2d5fold", "2fold", "3d5fold", "3fold", "4fold"]
    train,test,val = train_test_val_split("Train_Test_Val_split_IXI-T1.csv","DATA", "IXI-T1", folders,repeats = 2)

