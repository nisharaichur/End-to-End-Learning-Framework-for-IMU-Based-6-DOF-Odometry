import torch
import torch.nn as nn
from itertools import islice
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms

class Quaternion_Multiplicative_ErrorV2(torch.nn.Module):
    def __init__(self):
        print("QME optimized")
        super(Quaternion_Multiplicative_ErrorV2, self).__init__()
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
        #pro = self.hamilton_product(pred, true)
        pro = pred *  true
        img_part = pro[1:]
        #norm = np.linalg.norm(img_part, ord=1)

        norm = torch.norm(img_part, p=1)
        return 2 * norm

    def forward(self, pred, true):
        batch_size = pred.shape[0]
        return sum(self.qme(x, y) for x, y in zip(pred, true))/batch_size