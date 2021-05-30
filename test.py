import torch
import torch.nn as nn

a = torch.ones(2, 3)
b = a
c = torch.cat([a, b])
print(c)