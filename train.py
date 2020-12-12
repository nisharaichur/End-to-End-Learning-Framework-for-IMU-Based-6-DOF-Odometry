import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models
from itertools import islice
import torch.optim as optim
import configparser
import argparse
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from data_loading import IMUDataset
from imu_encoder import *
from training import *
import configparser
import argparse
import json

parser =  argparse.ArgumentParser()
parser.add_argument("--config", action="store", type=str, default="cfg.ini", help="provide config file")
args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)
print("**Loaded Configurations**\n")


dataset = config.get("hyper_prmts", "dataset")
beta = float(config.get("hyper_prmts", "beta"))
alpha = float(config.get("hyper_prmts", "alpha"))
learn_beta = config.get("hyper_prmts", "learn_beta")
loss_pos = config.get("hyper_prmts", "loss_pos")
loss_ori = config.get("hyper_prmts", "loss_ori")
optimizer = config.get("hyper_prmts", "optimizer")
batch_size = int(config.get("hyper_prmts", "batch_size"))
lr = float(config.get("hyper_prmts", "lr"))
SAVE_DIR = config.get("hyper_prmts", "save_dir")
n_epochs = int(config.get("hyper_prmts", "n_epochs"))
validation_freq = int(config.get("hyper_prmts", "validation_freq"))
save_check_point =  int(config.get("hyper_prmts", "save_check_point"))

print(f"dataset: {dataset}\n")
print(f"beta: {beta}\n")
print(f"alpha: {alpha}\n")
print(f"learn beta: {learn_beta}\n")
print(f"loss_pos: {loss_pos}\n")
print(f"loss_ori: {loss_ori}\n")
print(f"optimizetr: {optimizer}\n")
print(f"batch_size: {batch_size}\n")
print(f"lr: {lr}\n")
print(f"SAVE_DIR: {SAVE_DIR}\n")
print("\n")

transform = transforms.Compose([transforms.Resize((240, 376)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])


dataset = IMUDataset(dataset=dataset)
initial_pose = dataset.initial_pose

train_split = 0.8
dataset_size = len(dataset)
validation_split = .2
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_indices)
valid_sampler =  SequentialSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=train_sampler, pin_memory=True, num_workers=25)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=8, sampler=valid_sampler, pin_memory=True, num_workers=25)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('the device currently in use', device)

model = IMUEncoder(hidden_size=128)
model = model.to(device).double()

if learn_beta == "True":
    print("Learning the alpha and beta along with the weights of the network\n")
    criterion_relative = FusionCriterion_LearnParms(loss_pos=loss_pos, loss_ori=loss_ori, beta=beta, alpha=alpha).to(device=device)
    trainable_parms = [ {'params' : model.parameters()},
                        {'params' : criterion_relative.parameters()}]
else:
    criterion_relative = FusionCriterion(loss_pos=loss_pos, loss_ori=loss_ori, beta=beta, alpha=alpha).to(device=device)
    trainable_parms = model.parameters()

optimizer = optim.Adam(params=trainable_parms, lr=lr)

training_losses = dict()
validation_losses = dict()

training_losses = {"position_relative": [], "orientation_relative": []}
validation_losses = {"position_relative": [], "orientation_relative": []}



for e in range(n_epochs):
    absolute_pose_list_valid = dict()
    absolute_pose_list_train = dict()
    print("**Training**")
    losses_train = train(model, train_loader, criterion_relative, optimizer, device, e)
    training_losses["position_relative"].append(losses_train[0])
    training_losses["orientation_relative"].append(losses_train[1])
 
    if e % validation_freq==0:
        #plot_grad_flow(model.named_parameters(), epoch=e, save_dir=SAVE_DIR)
        print("**Validating**")
        
        losses_valid, pred_absolute_poses = validate(model, validation_loader, initial_pose, criterion_relative, device, e)
        validation_losses["position_relative"].append(losses_valid[0])
        validation_losses["orientation_relative"].append(losses_valid[1])
        
        if e % save_check_point == 0:
            absolute_pose_list_valid = pred_absolute_poses
        
    if e % save_check_point == 0:
        torch.save({
        "model_state_dict" : model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses" : training_losses,
        "validate_losses" : validation_losses,
        "predicted_absolute_poses_valid" : absolute_pose_list_valid,
            }, SAVE_DIR+f"/check_point{e}")

        
        #save_figures(validation_losses=validation_losses, training_losses=training_losses, epoch=e, absolute_poses=absolute_pose_list_valid)

