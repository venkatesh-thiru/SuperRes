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
    def __init__(self,nChannels,target_features,nDenseLayer,GrowthRate,KernelSize = 3):
        super(RDB,self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range (nDenseLayer):
            modules.append(make_dense(nChannels_,GrowthRate,kernel_size=KernelSize))
            nChannels_ += GrowthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_,target_features,kernel_size=1,padding=0,bias = False)
    def forward(self,x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class MultiScale(nn.Module):
    def __init__(self,nChannels,nDenseLayers,nInitFeat,GrowthRate,kernel_config = [7,5,3]):
        super(MultiScale,self).__init__()
        nChannels_ = nChannels
        nDenseLayers_ = nDenseLayers
        nInitFeat_ = nInitFeat
        GrowthRate_ = GrowthRate
        target_features = 6

        #First Convolution
        self.C1 = nn.Conv3d(nChannels_, nInitFeat_, kernel_size=kernel_config[0], padding=(kernel_config[0] - 1) // 2,bias=True)
        self.RDB1 = RDB(nInitFeat_,target_features, nDenseLayers_, GrowthRate_, kernel_config[0])

        self.C2 = nn.Conv3d(nChannels_, nInitFeat_, kernel_size=kernel_config[1], padding=(kernel_config[1] - 1) // 2,bias=True)
        self.RDB2 = RDB(nInitFeat_,target_features, nDenseLayers_, GrowthRate_, kernel_config[1])

        self.C3 = nn.Conv3d(nChannels_, nInitFeat_, kernel_size=kernel_config[2], padding=(kernel_config[2] - 1) // 2,bias=True)
        self.RDB3 = RDB(nInitFeat_,target_features, nDenseLayers_, GrowthRate_, kernel_config[2])

        self.FF_1X1 = nn.Conv3d(3*target_features,1,kernel_size=1,padding=0,bias=False)


    def forward(self,x):
        x1 = F.relu(self.C1(x))
        R1 = self.RDB1(x1)
        x2 = F.relu(self.C2(x))
        R2 = self.RDB2(x2)
        x3 = F.relu(self.C3(x))
        R3 = self.RDB3(x3)
        cats = torch.cat([R1,R2,R3],dim=1)

        FF = F.relu(self.FF_1X1(cats))

        return FF

if __name__ == '__main__':
    model = MultiScale(nChannels=1,nDenseLayers=6,nInitFeat=6,GrowthRate=12).cuda()
    dimensions = 24, 1, 32,32,32
    x = torch.rand(dimensions)
    x = x.cuda()
    out = model(x)
    print(model)
