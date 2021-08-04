import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from data.data_loader import Dataset_ETT_hour

a = [1, 2, 3, 4, 5]
b = [1, 2, 3, 4, 5]
c = [1, 2, 3, 4, 5]
d = [1, 2, 3, 4, 5]
for (A, B), C, D in zip(zip(a, b), c, d):
    print(A, B, C, D)
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
