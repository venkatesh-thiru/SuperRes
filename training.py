import torchio as tio
from torchio import AFFINE,DATA
from tqdm import tqdm
import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchio.transforms import Compose,ZNormalization,RescaleIntensity,RandomNoise
import statistics
import random
import DenseNetModel
import pytorch_ssim
from utils import train_test_val_split
from pLoss.perceptual_loss import PerceptualLoss
import multiprocessing
import UNetModel

torch.cuda.empty_cache()
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


def init_weights(m):
    if type(m) == nn.Conv3d:
        torch.nn.init.xavier_uniform_(m.weight)

# Initializing the model
# model = DenseNetModel.DenseNet(num_init_features=4,growth_rate=6,block_config=(6,6,6))
model = DenseNetModel.DenseNet(num_init_features=12,growth_rate=7,block_config=(6,6,6)).cuda()
model.apply(init_weights)


#Hyperparameters
learning_rate = 0.001
Epochs = 50
training_batch_size = 24
validation_batch_size = 6
patch_size = 48
samples_per_volume = 30
max_queue_length = 90


opt = optim.Adam(model.parameters(),lr=learning_rate)
loss_fn = pytorch_ssim.SSIM3D(window_size=11)


#setting up tensorboard

path = '/nfs1/ssaravan/code/runs'
training_name = "Trial_DenseNet_run_T2_FixedKernel_3".format(patch_size,samples_per_volume,Epochs,training_batch_size)
train_writer = SummaryWriter(os.path.join(path,"Densenets",training_name+"_training"))
validation_writer = SummaryWriter(os.path.join(path,"Densenets",training_name+"_validation"))

#Train Test Val split
training_subjects,test_subjects,validation_subjects = train_test_val_split()

#Data pipeline transform
training_transform = Compose([RescaleIntensity((0,1)),
                              RandomNoise(p=0.05)])
validation_transform = Compose([RescaleIntensity((0,1))])
test_transform = Compose([RescaleIntensity((0,1))])

#Generating subjects datasets
training_dataset = tio.SubjectsDataset(training_subjects,transform=training_transform)
validation_dataset = tio.SubjectsDataset(validation_subjects,transform=validation_transform)
test_dataset = tio.SubjectsDataset(test_subjects,transform=test_transform)


#Patcher initialization
patches_training_set = tio.Queue(
    subjects_dataset=training_dataset,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume,
    sampler=tio.sampler.UniformSampler(patch_size),
    # shuffle_subjects=True,
    # shuffle_patches=True,
)

patches_validation_set = tio.Queue(
    subjects_dataset=validation_dataset,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume*2,
    sampler=tio.sampler.UniformSampler(patch_size),
    # shuffle_subjects=False,
    # shuffle_patches=False,
)

#TrainLoader initialization
training_loader = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size, shuffle = True, num_workers=4)
validation_loader = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=validation_batch_size)


#Generating the tensorboard plots
def write_image(slice_list,epoch):
    print("writing image.......")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Epoch {}".format(epoch))
    ax[0].imshow(slice_list[0], interpolation='nearest', origin="lower", cmap="gray")
    ax[0].set_title("Original")
    ax[0].set_axis_off()
    ax[1].imshow(slice_list[1], interpolation='nearest', origin="lower", cmap="gray")
    ax[1].set_title("Reduced")
    ax[1].set_axis_off()
    ax[2].imshow(slice_list[2], interpolation='nearest', origin="lower", cmap="gray")
    ax[2].set_title("Predicted")
    ax[2].set_axis_off()
    train_writer.add_figure("comparison", fig, epoch)

def test_network(epoch):
    sample = random.choice(test_dataset)
    input_tensor = sample.compressed.data[0]
    patch_size = 64,64,64
    patch_overlap = 4,4,4
    model.eval()
    grid_sampler = tio.inference.GridSampler(sample,patch_size,patch_overlap)
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size = 2)
    aggregator = tio.inference.GridAggregator(grid_sampler,overlap_mode="average")
    with torch.no_grad():
        for batch in patch_loader:
            inputs = batch["compressed"][DATA].cuda()
            logits = model(inputs)
            location = batch[tio.LOCATION]
            aggregator.add_batch(logits,location)
    model.train()
    result = aggregator.get_output_tensor()
    original, compressed = torch.squeeze(sample.ground_truth["data"]), torch.squeeze(sample.compressed["data"])
    result = torch.squeeze(result)
    original,compressed,result = original.detach().cpu().numpy(),compressed.detach().cpu().numpy(),result.detach().cpu().numpy()
    slice_original = (original[:, :, int(original.shape[2] / 2)])
    slice_compressed = (compressed[:, :, int(compressed.shape[2] / 2)])
    slice_result = (result[:, :, int(result.shape[2] / 2)])
    slice_list = [slice_original.T,slice_compressed.T,slice_result.T]
    write_image(slice_list,epoch)

#Validation Loop
def validation_loop():
    print(("validating......."))
    overall_validation_loss = []
    model.eval()
    for batch in validation_loader:
        batch_actual = batch["ground_truth"][DATA].cuda()
        batch_compressed = batch["compressed"][DATA].cuda()
        with torch.no_grad():
            logit = model(batch_compressed)
        loss = loss_fn(logit, batch_actual)
        overall_validation_loss.append(loss.item())
    model.train()
    validation_loss = statistics.mean(overall_validation_loss)
    return validation_loss


#Training loop
steps = 0
old_validation_loss = 0
for epoch in range(Epochs):
    overall_training_loss = []
    for batch in tqdm(training_loader):
        steps += 1
        batch_actual = batch["ground_truth"][DATA].cuda()
        batch_compressed = batch["compressed"][DATA].cuda()
        logit = model(batch_compressed)
        loss =  -loss_fn(logit,batch_actual)
        opt.zero_grad()
        loss.backward()
        opt.step()
        overall_training_loss.append(-loss.item())
        if not steps % 100:
            training_loss = statistics.mean(overall_training_loss)
            train_writer.add_scalar("training_loss", training_loss, steps)
            test_network(steps)
            training_loss = statistics.mean(overall_training_loss)
            print("step {} : training_loss ===> {}".format(steps,training_loss))
            if not steps%1000 :
                validation_loss = validation_loop()
                validation_writer.add_scalar("validation_loss", validation_loss, steps)
                if (old_validation_loss == 0) or (old_validation_loss < validation_loss):
                    torch.save({'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': opt.state_dict(),
                                'loss': loss}, os.path.join("/nfs1/ssaravan/code/Models", training_name + ".pth"))
                    old_validation_loss = validation_loss
                    print("model_saved")