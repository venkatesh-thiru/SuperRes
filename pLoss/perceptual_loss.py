import math
import torch
import torch.nn as nn
import torchvision
from pLoss.VesselSeg_UNet3d_DeepSup import U_Net_DeepSup
from pytorch_ssim import SSIM3D
# from pytorch_msssim import SSIM,MS_SSIM

class PerceptualLoss(nn.Module):
    def __init__(self,n_level = math.inf,Loss_type="L1"):
        super(PerceptualLoss, self).__init__()
        blocks = []
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model = U_Net_DeepSup().cuda()
        model = nn.DataParallel(model)
        # chk = torch.load(r"pLoss/VesselSeg_UNet3d_DeepSup.pth",map_location=self.device)
        chk = torch.load(r"pLoss/VesselSeg_UNet3d_DeepSup.pth")
        model.load_state_dict(chk["state_dict"])
        blocks.append(model.Conv1.conv.eval())
        if n_level >= 2:
            blocks.append(
                nn.Sequential(
                    model.Maxpool1.eval(),
                    model.Conv2.conv.eval()
                )
            )
        if n_level >= 3:
            blocks.append(
                nn.Sequential(
                    model.Maxpool2.eval(),
                    model.Conv3.conv.eval()
                )
            )
        if n_level >= 4:
            blocks.append(
                nn.Sequential(
                    model.Maxpool3.eval(),
                    model.Conv4.conv.eval()
                )
            )
        if n_level >= 5:
            blocks.append(
                nn.Sequential(
                    model.Maxpool4.eval(),
                    model.Conv5.conv.eval()
                )
            )
        for bl in blocks:
            for parms in bl:
                parms.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        if Loss_type == "SSIM3D":
            self.loss_func = SSIM3D(window_size=11).to(self.device)
            # self.loss_func = SSIM3D(window_size=11)
        elif Loss_type == "L1":
            self.loss_func = torch.nn.functional.l1_loss


    def forward(self,input,target):
        loss = 0.0
        x,y = input,target
        for block in self.blocks:
            x=block(x)
            y=block(y)
            loss += self.loss_func(x,y)

        return loss

if __name__ == "__main__":
        x = PerceptualLoss().cuda()
        a = torch.rand(12, 1, 32, 32, 32).cuda()
        b = torch.rand(12, 1, 32, 32, 32).cuda()
        l = x(a, b)
        print(l)
