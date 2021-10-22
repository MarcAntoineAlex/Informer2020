import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from data.data_loader import Dataset_ETT_hour
import time

a = torch.ones(3, 4, 5)
b = torch.zeros(3, 4, 5)
a[:, 1, :] += a[:, 0, :]
a[:, 2, :] = a[:, 1, :]
print(a)
# data_set = Dataset_ETT_hour(
#             root_path="/Users/marc-antoine/Documents/Github/ETDataset/ETT-small",
#             data_path='ETTh1.csv',
#             flag='train',
#             size=[96, 48, 24],
#             features="M",
#             target='OT',
#             inverse=False,
#             timeenc=1,
#             freq='h',
#             cols=None
#         )
