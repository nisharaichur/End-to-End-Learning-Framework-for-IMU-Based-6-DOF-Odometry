import torch
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R
from matplotlib import lines
from torch.utils.data import Dataset, DataLoader
from util_funcs import *


def return_losses(output, poses):
    total_error = np.array([get_error(x[:3], x[3:], y[:3], y[3:]) for x, y in zip(output.detach().cpu().numpy(), poses.detach().cpu().numpy())])
    translation_error = np.mean(total_error[:, 0])
    rotational_error = np.mean(total_error[:, 1])
    return [translation_error, rotational_error]

def train(model, data_loader, criterion_relative, optimizer, device, epoch):
    model.train()
    start_time = time.time()
    rel_translation_error_sum = 0
    rel_orientation_error_sum = 0
    total_loss_sum = 0
    
    for idx, data in enumerate(data_loader):
        batch_imu = data["imu"].to(device)
        batch_relative_poses = data["relative_pose"].to(device)

        rel_output = model(batch_imu.double())

        total_loss =  criterion_relative(rel_output, batch_relative_poses)
        total_loss_sum = total_loss_sum + total_loss.detach().cpu().numpy()
        rel_translation_error, rel_rotational_error = return_losses(rel_output, batch_relative_poses)

        rel_translation_error_sum += rel_translation_error
        rel_orientation_error_sum += rel_rotational_error

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    time_taken = time.time() - start_time
    losses = [rel_translation_error_sum/idx, rel_orientation_error_sum/idx]
    print("Epoch: {} \t Time taken: {:.3f} \t Total_loss: {} \t  Relative_Pose_TrErro: {} \t Relative_Pose_RoError: {}".format(epoch, time_taken,  total_loss_sum/idx, rel_translation_error_sum/idx, rel_orientation_error_sum/idx))
    return  losses


def validate(model, data_loader, initial_pose, criterion_relative, device, epoch):
    model.eval()
    start_time = time.time()

    rel_translation_error_sum = 0
    rel_orientation_error_sum = 0
    total_loss_sum = 0
    relative_poses = []
    with torch.no_grad():
        for idx, data in enumerate(data_loader):
            batch_relative_poses = data["relative_pose"].to(device)
            batch_imu = data["imu"].to(device)
            rel_output = model(batch_imu.double())
            total_loss = criterion_relative(rel_output, batch_relative_poses)
            total_loss_sum = total_loss_sum + total_loss.cpu().numpy()
            relative_poses.append(rel_output.cpu())

            rel_translation_error, rel_rotational_error = return_losses(rel_output, batch_relative_poses)

            rel_translation_error_sum += rel_translation_error
            rel_orientation_error_sum += rel_rotational_error

        time_taken = time.time() - start_time

        losses = [rel_translation_error_sum/idx, rel_orientation_error_sum/idx]
        pred_realtive_poses = torch.cat(relative_poses, dim=0)
        pred_absolute_poses = return_absolute_poses(initial_pose, pred_realtive_poses)
        print("Epoch: {} \t Time taken: {:.3f} \t Total_loss: {}  \t Relative_Pose_TrErro: {} \t Relative_Pose_RoError: {}".format(epoch, time_taken, total_loss_sum/idx, rel_translation_error_sum/idx, rel_orientation_error_sum/idx))
        return  losses, pred_absolute_poses

def get_error(posx, posq, actualx, actualq):
    q1 = actualq / np.linalg.norm(actualq)
    #q1 = actualq
    q2 = posq / np.linalg.norm(posq)
    d = abs(np.sum(np.multiply(q1, q2)))
    theta = 2 * np.arccos(d) * 180 / np.pi
    errx = np.linalg.norm(actualx - posx, ord=1)
    return [errx, theta]

def return_absolute_poses(initial_pose, relative_poses):
    final_poses = trajectory_poses(relative_poses.numpy(), initial_pose)
    return final_poses

def plot_grad_flow(named_parameters, epoch, save_dir):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.figure(figsize=(200, 100))
    plt.plot(ave_grads, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k")
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=35)
    plt.yticks(fontsize=40)
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers", fontsize=40)
    plt.ylabel("average gradient", fontsize=40)
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(save_dir+f"/grads/gradient_flow{epoch}.png")
    plt.close()

"""def save_figures(validation_losses, training_losses, epoch, absolute_poses):

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
    for i in range(2):
        ax = np.ravel(ax[i])
        ax[0].plot()"""

