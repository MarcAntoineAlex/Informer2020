import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.ones(2, 3)
a[1][1] = float('-inf')
print(a)
print(F.softmax(a, dim=-1))