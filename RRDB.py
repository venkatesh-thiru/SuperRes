import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class make_dense(nn.Module):
    def __init__(self,nChannels,GrowthRate,kernel_size=3):
        super(make_dense,self).__init__()
        self.conv = nn.Conv3d(nChannels,GrowthRate,kernel_size=kernel_size,padding=(kernel_size-1)//2,bias=True)
        # self.norm = nn.BatchNorm3d(nChannels)
    def forward(self,x):
        # out = self.norm(x)
        out = F.relu(self.conv(x))
        out = torch.cat([x,out],dim=1)
        return out


class RDB(nn.Module):
    def __init__(self,nChannels,nDenseLayer,GrowthRate,KernelSize = 3):
        super(RDB,self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range (nDenseLayer):
            modules.append(make_dense(nChannels_,GrowthRate,kernel_size=KernelSize))
            nChannels_ += GrowthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_,nChannels,kernel_size=1,padding=0,bias = False)
    def forward(self,x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class RRDB(nn.Module):
    def __init__(self,nChannels,nDenseLayers,nInitFeat,GrowthRate,featureFusion=True,kernel_config = [7,7,5,3]):
        super(RRDB,self).__init__()
        nChannels_ = nChannels
        nDenseLayers_ = nDenseLayers
        nInitFeat_ = nInitFeat
        GrowthRate_ = GrowthRate
        self.featureFusion = featureFusion

        #First Convolution
        self.C1 = nn.Conv3d(nChannels_,nInitFeat_,kernel_size=kernel_config[0],padding=(kernel_config[0]-1)//2,bias=True)

        # Initialize RDB
        if self.featureFusion:
            self.RDB1 = RDB(nInitFeat_,nDenseLayers_,GrowthRate_,kernel_config[1])
            self.RDB2 = RDB(nInitFeat_, nDenseLayers_, GrowthRate_,kernel_config[2])
            self.RDB3 = RDB(nInitFeat_*2, nDenseLayers_, GrowthRate_,kernel_config[3])
            self.FF_1X1 = nn.Conv3d(nInitFeat_ * 4, 1, kernel_size=1, padding=0, bias=True)
        else:
            self.RDB1 = RDB(nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[1])
            self.RDB2 = RDB(nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[2])
            self.RDB3 = RDB(nInitFeat_, nDenseLayers_, GrowthRate_, kernel_config[3])
            self.FF_1X1 = nn.Conv3d(nInitFeat_, 1, kernel_size=1, padding=0, bias=True)


        # Feature Fusion


        # self.FF_3X3 = nn.Conv3d(nInitFeat_,nInitFeat_,kernel_size=3,padding=1,bias=True)

        # self.final_layer = nn.Conv3d(nInitFeat_,nChannels_,kernel_size=1,padding=0,bias=False)

    def forward(self,x):
        First = F.relu(self.C1(x))
        R_1 = self.RDB1(First)
        R_2 = self.RDB2(R_1)
        if self.featureFusion:
            FF1 = torch.cat([R_1,R_2],dim=1)
            R_3 = self.RDB3(FF1)
            FF2 = torch.cat([R_1, R_2, R_3], dim=1)
            FF1X1 = self.FF_1X1(FF2)
        else:
            R_3 = self.RDB3(R_2)
            FF1X1 = self.FF_1X1(R_3)

        # FF2 = torch.cat([R_1,R_2,R_3],dim=1)
        # FF1X1 = self.FF_1X1(FF2)
        # FF3X3 = self.FF_3X3(FF1X1)
        # output = self.final_layer(FF3X3)

        return FF1X1

if __name__ == '__main__':
    model = RRDB(nChannels=1,nDenseLayers=6,nInitFeat=6,GrowthRate=12,featureFusion=True).cuda()
    dimensions = 1, 1, 64, 64, 64
    x = torch.rand(dimensions)
    x = x.cuda()
    out = model(x)
