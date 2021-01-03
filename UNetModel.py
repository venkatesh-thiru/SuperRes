import torch
import torch.nn as nn


def conv3D(in_channel,filters,activation):
    return (nn.Sequential(
        nn.Conv3d(in_channel,filters,kernel_size= 3,stride=1,padding=1),
        nn.BatchNorm3d(filters),
        activation,
        nn.Conv3d(filters, filters, kernel_size=3, stride=1, padding = 1),
        nn.BatchNorm3d(filters)
    ))

def convTrans3D(in_channel,filters,activation):
    return (nn.Sequential(
        nn.ConvTranspose3d(in_channel,filters,kernel_size=3,stride=2,padding=1,output_padding=1),
        nn.BatchNorm3d(filters),
        activation
    ))

def maxPool3D():
    return nn.MaxPool3d(kernel_size=2,stride=2,padding=0)

class Unet(nn.Module):
    def __init__(self,in_channel,out_channel,filters):
        super(Unet,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.filters = filters
        activation = nn.PReLU()

        # Defining Encoder Block

        self.encoder_1 = conv3D(in_channel,filters,activation)
        self.pool_1 = maxPool3D()
        self.encoder_2 = conv3D(filters,filters*2,activation)
        self.pool_2 = maxPool3D()
        self.encoder_3 = conv3D(2*filters,4*filters,activation)
        self.pool_3 = maxPool3D()
        self.encoder_4 = conv3D(4*filters,8*filters,activation)
        self.pool_4 = maxPool3D()
        self.encoder_5 = conv3D(8*filters,16*filters,activation)
        self.pool_5 = maxPool3D()

        # Latent space representation

        self.latentSpace = conv3D(16 * filters, 32 * filters, activation)

        # Decoder Block

        self.decoder_1 = convTrans3D(32*filters,32*filters,activation)
        self.decoder_up_1 = conv3D (48*filters,16*filters,activation)
        self.decoder_2 = convTrans3D(16 * filters, 16 * filters, activation)
        self.decoder_up_2 = conv3D(24 * filters, 8 * filters, activation)
        self.decoder_3 = convTrans3D(8 * filters, 8 * filters, activation)
        self.decoder_up_3 = conv3D(12 * filters, 4 * filters, activation)
        self.decoder_4 = convTrans3D(4 * filters, 4 * filters, activation)
        self.decoder_up_4 = conv3D(6 * filters, 2 * filters, activation)
        self.decoder_5 = convTrans3D(2 * filters, 2 * filters, activation)
        self.decoder_up_5 = conv3D(3 * filters, filters, activation)

        # target

        self.out = conv3D(filters,out_channel,activation)

    def forward(self, x):

            down_1 = self.encoder_1(x)
            pool_1 = self.pool_1(down_1)



            down_2 = self.encoder_2(pool_1)
            pool_2 = self.pool_2(down_2)


            down_3 = self.encoder_3(pool_2)
            pool_3 = self.pool_3(down_3)

            down_4 = self.encoder_4(pool_3)
            pool_4 = self.pool_4(down_4)


            down_5 = self.encoder_5(pool_4)
            pool_5 = self.pool_5(down_5)


            latent = self.latentSpace(pool_5)


            trans_1 = self.decoder_1(latent)
            concat_1 = torch.cat([trans_1,down_5],dim=1)
            up_1 = self.decoder_up_1(concat_1)


            trans_2 = self.decoder_2(up_1)
            concat_2 = torch.cat([trans_2, down_4], dim=1)
            up_2 = self.decoder_up_2(concat_2)


            trans_3 = self.decoder_3(up_2)
            concat_3 = torch.cat([trans_3, down_3], dim=1)
            up_3 = self.decoder_up_3(concat_3)


            trans_4 = self.decoder_4(up_3)
            concat_4 = torch.cat([trans_4, down_2], dim=1)
            up_4 = self.decoder_up_4(concat_4)


            trans_5 = self.decoder_5(up_4)
            concat_5 = torch.cat([trans_5, down_1], dim=1)
            up_5 = self.decoder_up_5(concat_5)


            out = self.out(up_5)


            return out

if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dimensions = 20,1,64,64,64
    x = torch.rand(dimensions)
    x = x.to(device)
    model = Unet(1,1,8)
    print(model)
    model = model.to(device)
    out = model(x)

    print(out.shape)