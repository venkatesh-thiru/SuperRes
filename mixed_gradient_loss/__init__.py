import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
# from pytorch_msssim import ssim,ms_ssim
from torch.nn import L1Loss,MSELoss
from pytorch_ssim import ssim3D


def _custom_sobel(shape, axis):
    """
    https://stackoverflow.com/questions/9567882/sobel-filter-kernel-of-large-size
    shape must be odd: eg. (5,5)
    axis is the direction, with 0 to positive x and 1 to positive y
    """
    kernel = np.zeros(shape)
    p = [(k,j,i) for j in range(shape[0])
           for i in range(shape[1])
         for k in range(shape[2])
           if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2. and k == (shape[2]-1)/2.)]


    for k, j, i in p:
        k_ = int(k - (shape[2] -1)/2.)
        j_ = int(j - (shape[0] -1)/2.)
        i_ = int(i - (shape[1] -1)/2.)
        kernel[k,j,i] = (i_ if axis==0 else j_ if axis == 1 else k_)/float(i_*i_ + j_*j_ + k_*k_)
    return torch.tensor(kernel,requires_grad=False).unsqueeze(0).unsqueeze(0)

def _generate_edges(image,filterx,filtery,filterz):
    kernel_size = filterx.shape[-1]
    dx = F.conv3d(image, weight=filterx, padding=(kernel_size - 1) // 2, stride=1)
    dy = F.conv3d(image, weight=filtery, padding=(kernel_size - 1) // 2, stride=1)
    dz = F.conv3d(image, weight=filterz, padding=(kernel_size - 1) // 2, stride=1)

    edge = dx**2 + dy**2 + dz**2

    return edge


class mixed_gradient_loss(nn.Module):
    def __init__(self,sobel_size,loss_function='L1'):
        super(mixed_gradient_loss, self).__init__()
        self._filterx = _custom_sobel([sobel_size, sobel_size, sobel_size], axis=0)
        self._filtery = _custom_sobel([sobel_size, sobel_size, sobel_size], axis=1)
        self._filterz = _custom_sobel([sobel_size, sobel_size, sobel_size], axis=2)
        self.loss_function = loss_function

    def forward(self,img1,img2):

        if self._filterx.data.type() == img1.data.type():
            filterx = self._filterx
            filtery = self._filtery
            filterz = self._filterz
        else:
            filterx = self._filterx
            filtery = self._filtery
            filterz = self._filterz
            if img1.is_cuda:
                filterx = filterx.cuda(img1.get_device())
                filtery = filtery.cuda(img1.get_device())
                filterz = filterz.cuda(img1.get_device())
            filterx = filterx.type_as(img1)
            filtery = filtery.type_as(img1)
            filterz = filterz.type_as(img1)

        edge1 = _generate_edges(img1, filterx, filtery, filterz)
        edge2 = _generate_edges(img2, filterx, filtery, filterz)

        if self.loss_function == 'L1':
            return L1Loss(edge1,edge2)
        elif self.loss_function == 'SSIM':
            return ssim3D(edge1,edge2)
        # elif self.loss_function == 'MS-SSIM':       # todo: find a workaround for the 4 factor downsampling
        #     return ms_ssim(edge1,edge2)
        elif self.loss_function == 'MSE':
            return MSELoss(edge1,edge2)


if __name__ == '__main__':
    loss_fn = mixed_gradient_loss(3,loss_function='SSIM')
    x1 = torch.randn([1, 1, 64, 64, 64]).cuda()
    x2 = torch.randn([1, 1, 64, 64, 64]).cuda()
    loss = loss_fn(x1,x1)

    print(loss)