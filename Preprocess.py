import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import scipy
from scipy.ndimage import zoom
from fsl.data.image import Image
from fsl.utils.image import resample
import torchio as tio
import glob
from sklearn.model_selection import train_test_split
import argparse


def FFT_compression(myimg,scale_factor,ifzoom):
    '''
       Helper Function to compress images
       This method takes in the images and compresses them to the given scale factor
       uses the FSLPy library's resample function for compression. It also performs a nearest interpolation on the
       compressed image to make the compressed images ideal for training

       parameters:
       myimg(FSL Image)   : the image as FSL Image object
       scale_factor(List) : list of scale resolution along every dimensions to compress
       ifzoom             : Set false if the nearest interpolation is not required
    '''
    resample_img = resample.resampleToPixdims(myimg, scale_factor)
    new_affine = resample_img[1]
    if ifzoom:
        zoom_factor = np.array(myimg.shape) / np.array(resample_img[0].shape)
        zoomed = zoom(resample_img[0], zoom_factor, mode='nearest')
        return zoomed, new_affine
    else:
        return resample_img[0], new_affine

def write_data(compressed,scan,new_affine,scale_factor,dataset):
    '''
    Helper function to write compressed image as a nifti file
    '''
    path_name = "Compressed_{}x{}x{}_nozoom".format(scale_factor[0],scale_factor[1],scale_factor[2])
    target_dir = os.path.join(dataset,path_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    compressed_img = nib.Nifti1Image(compressed.astype(np.int16),new_affine)
    compressed_img.to_filename(os.path.join(target_dir, scan))

def prepare_datasets(scale_factor,dataset = 'IXI-T1',path = "Actual_Images",ifzoom=True):
    '''
    Execute this method to prepare the dataset for training. The method reads ground truth images as FSL Image objects
    compresses them and then writes them into a new directory
    PS: it is advisable to have the following folder structure
        main
        |_____IXI-T1(Scan Types)
              |____Actual_Images
    :param scale_factor: The target resolution as list
    :param dataset: Name of the dataset
    :param path: The subdirectory name with the ground truth images
    :param ifzoom: set false if a NN interpolation is not required
    :return: None
    '''
    data_dir = os.path.join(dataset,path)
    scans = os.listdir(data_dir)
    for scan in tqdm(scans):
        try:
            myimg = Image(os.path.join(data_dir, scan))
            compressed,new_affine = FFT_compression(myimg,scale_factor,ifzoom)
            compressed = compressed.astype(np.int16)
            write_data(compressed,scan,new_affine,scale_factor,dataset)
        except:
            print("{} invalid input".format(scan))
    print("write finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor",nargs="+")
    parser.add_argument("--dataset",default="IXI-T1")
    parser.add_argument("--path",default="Actual_Images")
    parser.add_argument("--ifzoom",default=True,type=bool)
    value = parser.parse_args()
    if len(value.scale_factor) == 3:
        scale_factor = [float(x) for x in value.scale_factor]
        dataset = value.dataset
        path = value.path
        ifzoom = value.ifzoom
        prepare_datasets(scale_factor,dataset,path,ifzoom)
    else:
        print("Give resolution across all dimensions")