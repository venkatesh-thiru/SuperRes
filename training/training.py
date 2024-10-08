import torchio as tio
from torchio import AFFINE,DATA
import torch
import os
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchio.transforms import Compose,ZNormalization,RescaleIntensity,RandomNoise
import statistics
import random
from utils import train_test_val_split
from losses.pLoss.perceptual_loss import PerceptualLoss
from tqdm import tqdm
import losses.pytorch_ssim as pytorch_ssim
import wandb
from wandb import AlertLevel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from math import ceil, floor
from torch.cuda.amp import autocast,GradScaler
from Evaluation import load_model_weights

torch.manual_seed(42)

torch.cuda.empty_cache()
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


def init_weights(m):
    if type(m) == nn.Conv3d:
        torch.nn.init.xavier_uniform_(m.weight)

# Initializing the model
# model = DenseNetModel.DenseNet(num_init_features=4,growth_rate=6,block_config=(6,6,6))
# model = DenseNetModel.DenseNet(num_init_features=12,growth_rate=7,block_config=(6,6,6)).cuda()
# model = RRDB(nChannels=1,nDenseLayers=6,nInitFeat=6,GrowthRate=12,featureFusion=True,kernel_config=[3,3,3,3]).cuda()
# model = MultiScale(nChannels=1,nDenseLayers=6,nInitFeat=6,GrowthRate=12).cuda()

# model = ShuffleUNet().cuda()



#Hyperparameters
hyper_batch_size = 22
loss_function = "L1"
model_kind = "RRDB"
image_intensity = "IXI-T1"
Epochs = 50
training_batch_size = 22
validation_batch_size = 22
patch_size = 64
samples_per_volume = 60
max_queue_length = 120
accumulation_steps = ceil(hyper_batch_size/training_batch_size)

if loss_function == 'SSIM':
    loss_fn = pytorch_ssim.SSIM3D(window_size=11).cuda()
    learning_rate = 0.0001
elif loss_function == 'L1':
    loss_fn = torch.nn.L1Loss()
    learning_rate = 0.0001
elif loss_function == 'perceptual_SSIM':
    loss_fn = PerceptualLoss(Loss_type="SSIM3D")
    learning_rate = 0.0001
elif loss_function == 'perceptual_L1':
    loss_fn = PerceptualLoss(Loss_type="L1")
    learning_rate = 0.0001


#Train Test Val split
'''
Do not forget to change things here
'''
code_path = "/nfs1/cheepina/code"
data_dir = "DATA"
data_path = os.path.join(code_path,data_dir)
csv_path = os.path.join(code_path,"Train_Test_Val_split_IXI-T1.csv")

folders = ["2d5fold", "2fold", "3d5fold", "3fold", "4fold"]
training_subjects,test_subjects,validation_subjects = train_test_val_split(csv_path,path =  data_path, intensity = image_intensity, folders =folders, repeats=2)
# training_subjects,test_subjects,validation_subjects = train_test_val_split(csv_path,path = data_path,intensity="IXI-T1",fold=fold)

#setting up tensorboard

path = 'runs'
training_name = f"mayura_RRDB_{loss_function}"
train_writer = SummaryWriter(os.path.join(path,"RRDB",training_name+"_training"))
validation_writer = SummaryWriter(os.path.join(path,"RRDB",training_name+"_validation"))

wandb.init(project="personals",name=training_name ,config={
    "learning_rate": learning_rate,
    "architecture": model_kind,
    "dataset": image_intensity,
    "Epochs" : Epochs,
})
wandb.alert(title = f"Training Began ",
            level = AlertLevel.INFO,
            text = f"GPU Started executing job ==> {training_name}")


model = load_model_weights(kind = model_kind)
model.apply(init_weights)
opt = optim.Adam(model.parameters(),lr=learning_rate)
scheduler = ReduceLROnPlateau(opt,'min',patience=10)
scaler = GradScaler()

#Data pipeline transform
training_transform = Compose([RescaleIntensity((0,1))])
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
    shuffle_subjects=True,
    shuffle_patches=True,
)

patches_validation_set = tio.Queue(
    subjects_dataset=validation_dataset,
    max_length=max_queue_length,
    samples_per_volume=samples_per_volume*2,
    sampler=tio.sampler.UniformSampler(patch_size),
    shuffle_subjects=False,
    shuffle_patches=False,
)

#TrainLoader initialization
training_loader = torch.utils.data.DataLoader(
    patches_training_set, batch_size=training_batch_size)
validation_loader = torch.utils.data.DataLoader(
    patches_validation_set, batch_size=validation_batch_size)

#Generating the tensorboard plots
def write_image(slice_list,epoch,space_dict):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5),dpi = 80,sharex=True, sharey=True)
    fig.suptitle("Epoch {}".format(epoch))

    ax[0].imshow(slice_list[0], interpolation='nearest', origin="lower", cmap="gray")
    ax[0].set_title("Original \n {}".format(space_dict["Actual_Image"]))
    ax[0].set_axis_off()

    ax[1].imshow(slice_list[1], interpolation='nearest', origin="lower", cmap="gray")
    ax[1].set_title("Downsampled \n {}".format(space_dict["Downsampled"]))
    ax[1].set_axis_off()

    ax[2].imshow(slice_list[2], interpolation='nearest', origin="lower", cmap="gray")
    ax[2].set_title("Interpolated")
    ax[2].set_axis_off()

    ax[3].imshow(slice_list[3], interpolation='nearest', origin="lower", cmap="gray")
    ax[3].set_title("Result")
    ax[3].set_axis_off()

    train_writer.add_figure("comparison", fig, epoch)
    wandb.log({f"Epoch {epoch}": fig})
    print("figure added")

def test_network(epoch):
    sample = random.choice(test_dataset)
    patch_size = 64,64,64
    patch_overlap = 10,10,10
    model.eval()
    grid_sampler = tio.inference.GridSampler(sample,patch_size,patch_overlap)
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size = 2)
    aggregator = tio.inference.GridAggregator(grid_sampler,overlap_mode="average")
    with torch.no_grad():
        for batch in patch_loader:
            inputs = batch["interpolated"][DATA].cuda()
            logits = model(inputs)
            location = batch[tio.LOCATION]
            aggregator.add_batch(logits,location)
    model.train()
    result = aggregator.get_output_tensor()
    fold = os.path.split(os.path.split(sample.interpolated.path)[0])[1]
    downsample_path = os.path.join(data_path,"IXI-T1","Compressed",fold)
    fname = sample.ground_truth.path.name
    file_path = os.path.join(downsample_path,fname)
    downsampled = tio.ScalarImage(file_path)
    o_scale,d_scale = sample.ground_truth.spacing,downsampled.spacing
    space_dict = {"Actual_Image": f"{round(o_scale[0], 2)}X{round(o_scale[1], 2)}X{round(o_scale[2], 2)}",
                  "Downsampled": f"{round(d_scale[0], 2)}X{round(d_scale[1], 2)}X{round(d_scale[2], 2)}"}
    original, interpolated,downsampled = torch.squeeze(sample.ground_truth["data"]), torch.squeeze(sample.interpolated["data"]), torch.squeeze(downsampled["data"])
    result = torch.squeeze(result)
    original,interpolated,result,downsampled = original.detach().cpu().numpy(),interpolated.detach().cpu().numpy(),result.detach().cpu().numpy(),downsampled.numpy()
    slice_original = (original[:, :, int(original.shape[2] / 2)])
    slice_interpolated = (interpolated[:, :, int(interpolated.shape[2] / 2)])
    slice_result = (result[:, :, int(result.shape[2] / 2)])
    slice_downsampled = (downsampled[:, :, int(downsampled.shape[2] / 2)])
    slice_list = [slice_original.T,slice_downsampled.T,slice_interpolated.T,slice_result.T]
    write_image(slice_list,epoch,space_dict)

#Validation Loop
def validation_loop():
    print(("validating......."))
    overall_validation_loss = []
    model.eval()
    for batch in validation_loader:
        batch_actual = batch["ground_truth"][DATA].cuda()
        batch_interpolated = batch["interpolated"][DATA].cuda()
        with torch.no_grad():
            logit = model(batch_interpolated)
        if loss_function == "SSIM":
            loss = 1-loss_fn(logit, batch_actual)
        else:
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
    opt.zero_grad()
    for i,batch in enumerate(tqdm(training_loader)):
        steps += 1
        batch_actual = batch["ground_truth"][DATA].cuda()
        batch_interpolated = batch["interpolated"][DATA].cuda()
        with autocast():
            logit = model(batch_interpolated)
            if loss_function == "SSIM":
                loss = 1-loss_fn(logit,batch_actual)
                overall_training_loss.append(loss.item())
            else:
                loss = loss_fn(logit, batch_actual)
                overall_training_loss.append(loss.item())

        scaler.scale(loss).backward()
        if not (i + 1) % accumulation_steps:
            opt.zero_grad()
            opt.step()



    #Train Loss and validation loss seggregation
    training_loss = statistics.mean(overall_training_loss)
    test_network(epoch)
    validation_loss = validation_loop()
    print(f"epoch {epoch} : training_loss ===> {training_loss} || Validation_loss ===> {validation_loss} \n")
    # wandb logging
    wandb.log({"validation_loss": validation_loss,"epoch" : epoch})
    wandb.log({"training_loss": training_loss,"epoch" : epoch})
    # tensorboard logging
    train_writer.add_scalar("training_loss", training_loss, epoch)
    validation_writer.add_scalar("validation_loss", validation_loss, epoch)
    # model saving
    # if loss_function == 'SSIM':
    #     if (old_validation_loss == 0) or (old_validation_loss < validation_loss):
    #         torch.save({'epoch': epoch,
    #                     'model_state_dict': model.state_dict(),
    #                     'optimizer_state_dict': opt.state_dict(),
    #                     'loss': loss}, os.path.join(r"Models", training_name + ".pth"))
    #         print("model_saved")
    # else:
    if (old_validation_loss == 0) or (old_validation_loss > validation_loss):
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': loss}, os.path.join(r"Models", training_name + ".pth"))
        print("model_saved")
    old_validation_loss = validation_loss