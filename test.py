import torch
import torch.nn as nn

a = torch.ones(2, 3)
a[1][2] = 2
print(a)
