import torch

model = torch.nn.Linear(1, 3)
data = torch.zeros(4, 1, 1, 1)
temp = model(data)

z = model(temp)

temp.sum().backward()
for p in model.parameters():
    print(p.grad)
z.sum().backward()
for p in model.parameters():
    print(p.grad)
