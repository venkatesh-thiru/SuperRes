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

Our implimented DenseNet model gave mean SSIM 0.89(median) results which surpassing all the classical interpollation methods.

![Classical interpollation methods vs DenseNet](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Example%20results/Example%204.png)
![Classical interpollation methods vs DenseNet](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Example%20results/Example3.png)



### Experimenting with differnt kernal sizes:

1. Our first experiment is to compare the performance DEnseNet with constant kernal size and varying kernal size.
We consider 3X3 kernels for constant kernal size model and 7X7, 5X5, 3X3 kernals for varying kernal size model. 
We observe that varying kernal size is gave best performance results of SSIM .90(median)

![constant and varying kernal size experiment](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/cnn.png)
![constant and varying kernal size experiment_value](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/Stats/CNN.PNG)


2. We have experimented with different kernel sizes on the DenseNet and found out that 
starting with a higher size and subsequently reducing the size of the kernel gave us best results.
This experiment is done considering 3 different resolution.

![Varying kernel size with  3 different scale factors experiment](https://github.com/v3nkyc0d3z/SuperRes/blob/master/Images/scale_factors.png)
 



