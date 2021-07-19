import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from data.data_loader import Dataset_ETT_hour

n = torch.randint(10, (5, 5))
m = torch.arange(10)[:, None, None]
a = 1.0
b = 2
l = nn.Linear(a, b)
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