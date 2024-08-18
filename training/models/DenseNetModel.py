
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DenseLayer(nn.Sequential):
    def __init__(self,filters,growth_rate,bottle_neck_size,kernel_size):
        super(DenseLayer,self).__init__()
        self.add_module("norm1",nn.BatchNorm3d(filters))
        self.add_module("relu1",nn.ReLU(inplace=True))
        self.add_module("conv1",nn.Conv3d(filters,bottle_neck_size*growth_rate,kernel_size=1,stride=1,padding=0))
        self.add_module("norm2",nn.BatchNorm3d(bottle_neck_size*growth_rate))
        self.add_module("relu2",nn.ReLU(inplace=True))
        self.add_module("conv2",nn.Conv3d(bottle_neck_size*growth_rate,growth_rate,kernel_size=kernel_size,stride=1,padding=(kernel_size-1)//2))


    def forward(self, input):
        new_features = super().forward(input)
        return torch.cat([input,new_features],1)


class DenseBlock(nn.Sequential):
    def __init__(self,num_layers,num_input_features,growth_rate,bottle_neck_size,kernel_size=3):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(num_input_features+i*growth_rate,growth_rate,bottle_neck_size,kernel_size)
            self.add_module('denselayer{}'.format(i+1),layer)


class Transition(nn.Sequential):
    def __init__(self,num_input_features,num_output_features):
        super(Transition, self).__init__()
        self.add_module("norm",nn.BatchNorm3d(num_input_features))
        self.add_module("relu",nn.ReLU(inplace=True))
        self.add_module("conv",nn.Conv3d(num_input_features,num_output_features,kernel_size=1,stride=1,padding=0))


class DenseNet(nn.Module):
    def __init__(self,n_input_channel=1,
                 growth_rate = 12,
                 block_config = (6,12,24,16),
                 num_init_features = 64,
                 bottle_neck_size = 4):
        super(DenseNet, self).__init__()
        self.features = [('conv',nn.Conv3d(n_input_channel,num_init_features,
                                           kernel_size=3,
                                           stride =1,
                                           padding=1,
                                           bias=False)),
                          ('norm1',nn.BatchNorm3d(num_init_features)),
                          ("relu",nn.ReLU(inplace=True))]

        self.features = nn.Sequential(OrderedDict(self.features))
        init_kernel_size = 3
        num_features = num_init_features
        for i,num_layers in enumerate(block_config):
            block = DenseBlock(num_layers=num_layers,
                               num_input_features=num_features,
                               bottle_neck_size=bottle_neck_size,
                               growth_rate=growth_rate,
                               kernel_size=init_kernel_size)
            self.features.add_module("denseblock{}".format(i+1),block)
            num_features = num_features+growth_rate*num_layers
            # init_kernel_size -= 2       #kernel size increases by 2 for each dense blocks
            if i != len(block_config)-1:
                transition = Transition(num_input_features=num_features,
                                        num_output_features=num_features//2)    #compression factor 0.5
                self.features.add_module("transition{}".format(i+1),transition)
                num_features = num_features//2

        self.features.add_module("final1x1",nn.Conv3d(num_features,1,kernel_size=1,stride=1,padding=0))

    def forward(self,x):
        features = self.features(x)
        out = F.relu(features,inplace= True)
        return out





if __name__ == '__main__':
    model = DenseNet(num_init_features=4,growth_rate=6,block_config=(6,6,6))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dimensions = 2, 1, 32,32,32
    x = torch.rand(dimensions)
    x = x.cuda()
    model = model.cuda()
    out = model(x)
    print(out.shape)
    print(model)