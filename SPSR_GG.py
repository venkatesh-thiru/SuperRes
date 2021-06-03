from RRDB import RRDB
from RRDB import RDB
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

def _generate_edges(image,sobel_size = 3):
    filterx = _custom_sobel([sobel_size, sobel_size, sobel_size], axis=0)
    filtery = _custom_sobel([sobel_size, sobel_size, sobel_size], axis=1)
    filterz = _custom_sobel([sobel_size, sobel_size, sobel_size], axis=2)

    if filterx.data.type() == image.data.type():
        filterx = filterx
        filtery = filtery
        filterz = filterz
    else:
        filterx = filterx
        filtery = filtery
        filterz = filterz
        if image.is_cuda:
            filterx = filterx.cuda(image.get_device())
            filtery = filtery.cuda(image.get_device())
            filterz = filterz.cuda(image.get_device())
        filterx = filterx.type_as(image)
        filtery = filtery.type_as(image)
        filterz = filterz.type_as(image)

    kernel_size = filterx.shape[-1]

    dx = F.conv3d(image, weight=filterx, padding=(kernel_size - 1) // 2, stride=1)
    dy = F.conv3d(image, weight=filtery, padding=(kernel_size - 1) // 2, stride=1)
    dz = F.conv3d(image, weight=filterz, padding=(kernel_size - 1) // 2, stride=1)

    edge = torch.sqrt(dx**2 + dy**2 + dz**2)

    return edge

class SPSR_GG(nn.Module):
    def __init__(self,nChannels,nDenseLayers,nInitFeat,GrowthRate,kernel_config = [3,3,3,3],sobel_size = 3):
        super(SPSR_GG,self).__init__()
        nChannels_ = nChannels
        nDenseLayers_ = nDenseLayers
        nInitFeat_ = nInitFeat
        GrowthRate_ = GrowthRate

        # SR_MODULES
        # First Convolution
        self.C1 = nn.Conv3d(nChannels_, nInitFeat_, kernel_size=kernel_config[0], padding=(kernel_config[0] - 1) // 2,
                            bias=True)
        # Initialize RDB
        self.RDB1 = RDB(nInitFeat_, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[1])
        # print(f"RDB1 =========================================== \n {self.RDB1}")
        self.RDB2 = RDB(nInitFeat_ * 2, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[2])
        # print(f"RDB2 =========================================== \n {self.RDB2}")
        self.RDB3 = RDB(nInitFeat_ * 3, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[3])

        # Gradient Modules

        # First Convolution
        self.C1_G = nn.Conv3d(nChannels_, nInitFeat_, kernel_size=kernel_config[0], padding=(kernel_config[0] - 1) // 2,
                            bias=True)
        # Initialize RDB
        self.RDB1_G = RDB(nInitFeat_, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[1])
        # print(f"RDB1 =========================================== \n {self.RDB1}")
        self.RDB2_G = RDB(nInitFeat_ * 2, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[2])
        # print(f"RDB2 =========================================== \n {self.RDB2}")
        self.RDB3_G = RDB(nInitFeat_ * 3, nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[3])

        # Final Fusion
        self.F1x1 = nn.Conv3d(nInitFeat_*8, 1, kernel_size=1, padding=0, bias=True)

        # Edge Fusion
        self.GF1x1 = nn.Conv3d(nInitFeat_*4, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, img1):

        edge = _generate_edges(img1)

        # SR Path
        First = F.relu(self.C1(img1))
        R_1 = self.RDB1(First)
        FF0 = torch.cat([First, R_1], dim=1)
        R_2 = self.RDB2(FF0)
        FF1 = torch.cat([First, R_1, R_2],dim = 1)
        R_3 = self.RDB3(FF1)

        # Edge Path
        First_G = F.relu(self.C1_G(edge))
        R_1_G = self.RDB1_G(First_G)
        FF0_G = torch.cat([First_G, R_1_G], dim=1)
        R_2_G = self.RDB2_G(FF0_G)
        FF1_G = torch.cat([First_G, R_1_G, R_2_G],dim = 1)
        R_3_G = self.RDB3_G(FF1_G)

        GFF = torch.cat([First,First_G,
                         R_1,R_1_G,
                         R_2,R_2_G,
                         R_3,R_3_G],dim = 1)

        FF1x1 = F.relu(self.F1x1(GFF))

        if self.training:
            edge_FF1x1 = F.relu(self.GF1x1(torch.cat([First_G, R_1_G, R_2_G, R_3_G],dim = 1)))
            return FF1x1,edge_FF1x1
        else:
            return FF1x1


if __name__ == '__main__':
    # model = RRDB(nChannels=1,nDenseLayers=6,nInitFeat=6,GrowthRate=12,featureFusion=True,kernel_config = [3,3,3,3]).cuda()
    model = SPSR_GG(nChannels=1,nDenseLayers=6,nInitFeat=6,GrowthRate=12,kernel_config = [3,3,3,3]).cuda()
    dimensions = 13, 1, 32,32,32

    x = torch.rand(dimensions)
    x = x.cuda()
    SR,edgeSR = model(x)
    print(model)
    print(SR.shape)
    print(edgeSR.shape)