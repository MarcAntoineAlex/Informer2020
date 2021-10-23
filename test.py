import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from data.data_loader import Dataset_ETT_hour
import time

batch_y = torch.randn(3, 4, 5)
origin_y = batch_y[:, -2:, :].detach()
y1 = batch_y[:, -2, :].detach()
my = torch.cat([batch_y[:, 0, :].unsqueeze(1), batch_y[:, :-1, :]], dim=1)
batch_y -= my

batch_y = batch_y[:, -2:, :]
batch_y[:, 0, :] = y1
for i in range(1, 2):
    batch_y[:, i, :] += batch_y[:, i - 1, :]
print(origin_y-batch_y)
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
