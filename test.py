import torch
import torch.nn as nn

a = torch.ones(2, 3)
a[1][1] = float('-inf')
print(a)
a = torch.softmax(a, -1)
print(a)