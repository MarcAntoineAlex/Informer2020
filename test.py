import torch
import torch.nn as nn

a = torch.ones(2, 3)
a += float('-inf')
print(a)
