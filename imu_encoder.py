import torch
import torch.nn as nn
from itertools import islice
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms

class IMUEncoder(torch.nn.Module):
    def __init__(self, hidden_size):
        print("IMU Encoder")
        super(IMUEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.convA1 = torch.nn.Conv1d(in_channels=3, out_channels=128, kernel_size=11)
        self.convA2 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=11)
        self.relu = torch.nn.ReLU()
        self.tanh = torch.nn.Tanh()
        self.convB1 = torch.nn.Conv1d(in_channels=3, out_channels=128, kernel_size=11)
        self.convB2 = torch.nn.Conv1d(in_channels=128, out_channels=128, kernel_size=11)
        self.max_pool1 = torch.nn.MaxPool1d(kernel_size=3)
        self.max_pool2 = torch.nn.MaxPool1d(kernel_size=3)
        self.lstm1 = torch.nn.LSTM(input_size=256, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=256, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.fc_pos = torch.nn.Linear(in_features=256, out_features=3)
        self.fc_ori = torch.nn.Linear(in_features=256, out_features=4)
     
    """def forward(self, input):
        ang_vel = input[:, :3, :]
        acc = input[:, 3:, :]
        ang_vel = self.relu(self.convA1(ang_vel))
        
        ang_vel = self.relu(self.convA2(ang_vel))
        
        acc = self.relu(self.convB1(acc))
        
        
        acc = self.relu(self.convB2(acc))
        
        ang_vel = self.max_pool1(ang_vel)
        
        acc = self.max_pool2(acc)
        
        x = torch.cat((ang_vel, acc), dim=1)
        
        x, _ = self.lstm1(x)
        
        x = self.tanh(x)
        x = F.dropout(x, 0.25)
        x, _ = self.lstm2(x)
        
        x = self.tanh(x)
        x = F.dropout(x, 0.25)
        direction1 = x[:, -1, :128]
        direction2 = x[:, 0, 128:]
        x = torch.cat((direction1, direction2), dim=1)
        
        pos = self.fc_pos(x)
       
        ori = self.fc_ori(x)
        
        return torch.cat((pos, ori), dim=1)"""
    def forward(self, input):
        
        # B x 6 x 200
        ang_vel = input[:, :3, :]
        acc = input[:, 3:, :]
        ang_vel = self.convA1(ang_vel)
        
        # B x 128 x 190
        ang_vel = self.convA2(ang_vel)
        
        # B x 128 x 180
        acc = self.convB1(acc)
        
        # B x 128 x 190
        acc = self.convB2(acc)
     
        # B x 128 x 180
        ang_vel = self.max_pool1(ang_vel)
        
        # B x 128 x 60
        acc = self.max_pool2(acc)
        
        # B x 128 x 60
        x = torch.cat((ang_vel, acc), dim=1)
        
        # B x 256 x 60
        x = x.reshape(-1, x.shape[-1], x.shape[1])
        
        # B x 60 x 256
        x, _ = self.lstm1(x)
        x = self.tanh(x)
        
        # B x 60 x 256
        x = F.dropout(x, 0.35)
        x, _ = self.lstm2(x)
        x = self.tanh(x)
        
        # B x 60 x 256
        x = x[:, -1, :]
        x = F.dropout(x, 0.35)
        
        # B x 256
        pos = self.fc_pos(x)
        
        # B x 3
        ori = self.fc_ori(x)
        
        # B x 4
        return torch.cat((pos, ori), dim=1)
   
class Quaternion_Multiplicative_Error(torch.nn.Module):
    def __init__(self):
        print("QME optimized")
        super(Quaternion_Multiplicative_Error, self).__init__()
        #self.conj = torch.nn.Parameter(torch.tensor([1,-1,-1,-1]), requires_grad=False)
        self.register_buffer("conj", torch.tensor([1,-1,-1,-1]))

    def hamilton_product(self, quat1, quat2):
        a1, b1, c1, d1 = quat1
        a2, b2, c2, d2 = quat2
        q1 = a1*a2- b1*b2 - c1*c2 -d1*d2
        q2 = a1*b2 + b1*a2 + c1*d2 - d1*c2
        q3 = a1*c2 - b1*d2 + c1*a2 + d1*b2
        q4 = a1*d2 + b1*c2 - c1*b2 + d1*a2
        return np.array([q1, q2, q3, q4])

    def qme(self, pred, true):
        true = torch.mul(true, self.conj)
        pro = self.hamilton_product(pred, true)
        #pro = pred *  true
        img_part = pro[1:]
        norm = np.linalg.norm(img_part, ord=1)
        #norm = torch.norm(img_part, p=1)
        return 2 * norm

    def forward(self, pred, true):
        batch_size = pred.shape[0]
        return sum(self.qme(x, y) for x, y in zip(pred, true))/batch_size
 
class FusionCriterion_LearnParms(torch.nn.Module):
    def __init__(self, loss_pos="L1Loss", loss_ori="QMELoss", alpha=0.0, beta=-3.0):
        super(FusionCriterion_LearnParms, self).__init__()
        self.loss_pos = self.select_loss(loss_pos)
        self.loss_ori = self.select_loss(loss_ori)
        self.alpha = torch.nn.Parameter(torch.tensor([alpha], dtype=torch.double), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor([beta], dtype=torch.double), requires_grad=True)
        
    def select_loss(self, loss):
        if loss == "L1Loss":
            print("Optimized using L1 Loss")
            return torch.nn.L1Loss()    
        elif loss == "MSELoss":
            print("Optimized using MSE Loss ")
            return torch.nn.MSELoss()
        else:
            print("Optimized using QME Loss")
            return Quaternion_Multiplicative_Error()

    def normalize_quaternion(self, q):
        norm = torch.norm(q, dim=1)
        """None helps to broadcast"""
        q_norm = torch.div(q, norm[:, None])
        return q_norm

    def forward(self, predicted, actual):
        position_loss = (torch.exp(-self.alpha) * self.loss_pos(predicted[:, :3], actual[:, :3])) + self.alpha
        orientation_loss = (torch.exp(-self.beta) * self.loss_ori(self.normalize_quaternion(predicted[:, 3:]), actual[:, 3:])) + self.beta
        total_loss =   position_loss + orientation_loss
        return total_loss


class FusionCriterion(torch.nn.Module):
    def __init__(self, loss_pos="L1Loss", loss_ori="QMELOSS", alpha=20, beta=10):
        super(FusionCriterion, self).__init__()
        self.loss_pos = self.select_loss(loss_pos)
        self.loss_ori = self.select_loss(loss_ori)
        self.alpha = alpha
        self.beta = beta
        
    def select_loss(self, loss):
        if loss == "L1Loss":
            print("Optimized using L1 Loss")
            return torch.nn.L1Loss()    
        elif loss == "MSELoss":
            print("Optimized using MSE Loss ")
            return torch.nn.MSELoss()
        else:
            print("Optimized using QME Loss")
            return Quaternion_Multiplicative_Error().to(device=torch.device('cuda'))

    def normalize_quaternion(self, q):
        norm = torch.norm(q, dim=1)
        """None helps to broadcast"""
        q_norm = torch.div(q, norm[:, None])
        return q_norm     

    def forward(self, predicted, actual):
        position_loss = self.loss_pos(predicted[:, :3], actual[:, :3]) 
        orientation_loss = self.loss_ori(self.normalize_quaternion(predicted[:, 3:]), actual[:, 3:]) 
        total_loss = self.alpha  *  position_loss + self.beta * orientation_loss
        return total_loss

