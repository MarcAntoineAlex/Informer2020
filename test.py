import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.ones(2, 3)
b = torch.ones(2, 3)
c = a*b.sum()
print(c)