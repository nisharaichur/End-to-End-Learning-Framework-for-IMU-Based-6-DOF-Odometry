import torch
import pandas as pd
import os
from scipy.spatial.transform import Rotation as R
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
from itertools import islice
from matplotlib import pyplot as plt
ROOT_DIR = "/data/beegfs/home/raichuna/thesis-fraunhofer-iis/Dataset"

def normalize_quaternion(q):
        """torch.norm has the default norm of 2 squaroot(sum of squares) """
        norm = torch.norm(q, dim=1)
        q_norm = torch.div(q, norm[:, None])
        return q_norm

def hamilton_product(quat1, quat2):
        a1, b1, c1, d1 = quat1
        a2, b2, c2, d2 = quat2
        q1 = a1*a2- b1*b2 - c1*c2 -d1*d2
        q2 = a1*b2 + b1*a2 + c1*d2 - d1*c2
        q3 = a1*c2 - b1*d2 + c1*a2 + d1*b2
        q4 = a1*d2 + b1*c2 - c1*b2 + d1*a2
        return np.array([q1, q2, q3, q4])

def trajectory_poses(poses_predicted, initial_pose):
    final_poses = np.empty((0,7))
    #for x in islice(poses_predicted, 200):
    for x in poses_predicted:
        delta_p = x[:3] 
        delta_q = x[3:] / np.linalg.norm(x[3:])
        pt_1 = initial_pose[:3]
        qt_1 = initial_pose[3:] / np.linalg.norm(initial_pose[3:])
        
        r = R.from_quat(qt_1)
        qt_1_mat = r.as_matrix()
        p =  pt_1 +  np.matmul(qt_1_mat, delta_p)
        q = hamilton_product(qt_1, delta_q)
        initial_pose = np.concatenate((p, q))
        
        final_poses = np.vstack((final_poses, initial_pose[np.newaxis, :]))
    return final_poses
 

def return_poses(dataset, path):
    check = torch.load(path, map_location=torch.device("cpu"))
    
    data_set = IMUDataset(dataset)
    loader = DataLoader(data_set, 32)
    relative = torch.cat(list(data["relative_pose"] for data in loader), dim=0)
    pose = pd.read_csv(ROOT_DIR + dataset + "/mav0/state_groundtruth_estimate0/data.csv").values

    initial_pose_true = pose[0, 1:8]
    poses_predicted = check["absolute_pose"][dataset].numpy()

    final_poses_true = trajectory_poses(relative.numpy(), initial_pose_true)
    final_poses = trajectory_poses(poses_predicted, initial_pose_true)

    return final_poses_true, final_poses