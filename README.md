# MRI-Super Resolution using 3D CNNs

Code for "Beyond Nyquist: A Comparative Analysis of 3D Deep Learning Models"


## Table of Content
* [Abstract](#abstract)
* [Installation](#installation)
* [Usage](#usage)
* [Pretrained models](#pre-trained-models)
* [Inferencing](#inferencing)
* [Feedbacks](#feedbacks)

## Abstract

High spatial resolution MRI produces abundant structural information, enabling highly
accurate clinical diagnosis and image-guided therapeutics. However, the acquisition of high spatial
resolution MRI data typically can come at the expense of less spatial coverage, lower signal-to-noise
ratio (SNR), and longer scan time due to physical, physiological and hardware limitations. In order
to overcome these limitations Super-resolution MRI deep learning based techniques can be utilised.
In this work, different state-of-the-art 3D convolution neural network models for super resolution
(RRDB, SPSR, UNet, UNet-MSS and ShuffleUNet) were compared for the super-resolution task with
the goal of finding the best model in terms of performance and robustness. The public IXI dataset
(only structural images) was used. Data were artificially downsampled to obtain lower resolution
spatial MRIs (downsampling factor varying from 8 to 64). When assessing performance using the
SSIM metric in the test set, all models performed well. In particular, regardless of the downsampling
factor, the UNet consistently obtained the top results. On the other hand, the SPSR model always
performed worse. In conclusion, UNet and UNet-MSS achieved overall top performances while
RRDB performed relatively poorly compared to the other models.

## Pretrained Models

The pretrained models have been uploaded to the huggingface hub and can be found under [SRMRI - 3D CNN pretrained model collections](https://huggingface.co/collections/venkatesh-thiru/srmri-3d-cnn-pretrained-model-collections-66c26c1dcb6aab077492fec3)

🚧🚧🚧🚧🚧🚧**UNDER CONSTRUCTION**🚧🚧🚧🚧🚧🚧