import numpy as np

import torch

a = torch.tensor([[1,0],[5,5],[6,6],[3,2]])
b = torch.tensor([[1,0],[5,5],[6,6],[1,4]])

w = torch.tensor([1,1])
ww = torch.tensor([[1,1],[1,1],[1,1],[1,1]])

c = torch.einsum('ij,j->i',[a,w])
torch.unsqueeze(c, 1)



a = torch.tensor([[87,87],[ 87, 87], [87,87], [87,87]])
b = torch.tensor([[78,78],[ 78, 78], [78,78], [78,78]])
x = torch.tensor([1, 2, 3, 4])
y = torch.tensor([4, 3, 2, 1])

print(a)
print(b)
mask = torch.ge(x,y)
print(mask)
#mask = torch.unsqueeze(mask, 0)
mask = mask.repeat([1,2])
print(mask)
mask = torch.reshape( mask, a.shape)
print(mask)


#c = torch.where( torch.ge(x,y), a,b)
c = torch.where( mask, a,b)

print(c)
