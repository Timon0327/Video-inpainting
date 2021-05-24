import torch
import numpy as np


def loss_chedistance(y_pre, y, m):
    # che_distance = np.max(np.abs(y_pre - y))
    # loss = np.max(m.mul((y_pre - y))) / (torch.norm(m, p=1))
    loss = torch.norm(m.mul(y_pre - y), p='inf') / (torch.norm(m, p='inf'))

    return loss

def loss_mandistance(y_pre, y, m):
    # che_distance = np.max(np.abs(y_pre - y))
    # loss = np.max(m.mul((y_pre - y))) / (torch.norm(m, p=1))
    loss = torch.norm(m.mul(y_pre - y), p=1) / (torch.norm(m, p=1))

    return loss

x = torch.rand(1, 3)
y = torch.rand(1, 3)
m = torch.rand(1, 3)
# print(x)
loss = loss_mandistance(x, y, m)
print(loss)