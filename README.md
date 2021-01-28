# SuperRes
Clean code of Super Resolution

All the results are obtained with input of 4x4x1.2 compressed images. 


The classical interpolation methods we have used are
1. Nearest neighbour
2. Bilinear
3. Bicubic

The result comaparision is as follows

![Classical interpollation methods](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Legacy%20.png)
![Classical interpollation methods values](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Stats/legacy.PNG)


From the results we can see that best result is SSIM .8(median) obtained from bicubic interpollation.

Our implemented DenseNet model gave mean SSIM 0.89(median) results which surpassing all the classical interpolation methods.

![Classical interpollation methods vs DenseNet](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Example%20results/Example%204.png)
![Classical interpollation methods vs DenseNet](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Example%20results/Example3.png)



### Experimenting with differnt kernal sizes:

1. Our first experiment is to compare the performance DenseNet with constant kernal size and varying kernal size.
We consider 3X3 kernels for constant kernal size model and 7X7, 5X5, 3X3 kernals for varying kernal size model. 
We observe that varying kernal size is gave best performance results of SSIM .90(median)

![constant and varying kernal size experiment](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/cnn.png)
![constant and varying kernal size experiment_value](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Stats/CNN.PNG)


2. We have experimented with different kernel sizes on the DenseNet and found out that 
starting with a higher size and subsequently reducing the size of the kernel gave us best results.
This experiment is done considering 3 different resolution.

![Varying kernel size with  3 different scale factors experiment](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/scale_factors.png)
 
 ### L1 loss function for IXI-T1 dataset
 The model was trained with the following hyperparameters for IXI-T1 dataset for the L1 loss function:
 learning_rate = 0.001
 Epochs = 50
 training_batch_size = 24
 validation_batch_size = 6
 patch_size = 48
 samples_per_volume = 30
 max_queue_length = 90
 
 The model was trained for 50 epochs and the output was as follows
![L1 loss function for IXI-T1 dataset (50 epochs)](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Example%20results/T1_L1_50epochs.png)
The training loss was 0.034913947040747316

 To remove the grid lines which appeared in the output, following changes were made to the hyperparameters:
 learning_rate = 0.00001
 Epochs = 200
 training_batch_size = 24
 validation_batch_size = 6
 patch_size = 48
 samples_per_volume = 40
 max_queue_length = 120
 
  The model was trained for 200 epochs and the output was as follows
  ![L1 loss function for IXI-T1 dataset (200 epochs)](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Example%20results/T1_L1_200epochs.png)
 The training loss was 0.0258860532194376
 
 ### Initial Experiments on IXI-T2 dataset
 
 ## Loss Function: SSIM
 
 learning_rate = 0.001
 Epochs = 50
 training_batch_size = 24
 validation_batch_size = 6
 patch_size = 48
 samples_per_volume = 30
 max_queue_length = 90
 
 # Training Output:
 
![SSIM-IXI-T2 dataset (50 epochs)](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Example%20results/T2_SSIM_50.jpg)

Training loss: 0.9105

## Loss Function: L1

 learning_rate = 0.00001
 Epochs = 200
 training_batch_size = 24
 validation_batch_size = 6
 patch_size = 48
 samples_per_volume = 60
 max_queue_length = 120
 
 # Training Output:
 
 ![L1-IXI-T2 dataset (50 epochs)](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Example%20results/T2_L1_200.jpg)
 
 Training loss: 0.021400894038379192
