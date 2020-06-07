import torch

x = torch.ones((5,6,3))

y = torch.zeros(x.shape)

print(torch.max(x,y))