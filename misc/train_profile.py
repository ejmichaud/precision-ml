import random
import time
from itertools import product, islice
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision

########### Set Device ############
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64
torch.set_default_dtype(dtype)
print("Using device: {}".format(device))

def loader_for_function(f, ranges, device=device, dtype=dtype):
    def get_batch(N=1000):
        xs = []
        ys = []
        for _ in range(N):   
            x = tuple(r1 + (r2 - r1) * np.random.rand() for r1, r2 in ranges)
            y = f(*x)
            xs.append(x)
            ys.append(y)
        xs = torch.tensor(xs).to(dtype).to(device)
        ys = torch.tensor(ys).unsqueeze(1).to(dtype).to(device)
        return xs, ys
    
    return get_batch

batch_loader = loader_for_function(lambda x: x, ranges=[(-1, 1)])
batch_size = 100000
steps = 100

width = 200
mlp = nn.Sequential(
    nn.Linear(1, width),
    nn.Tanh(),
    nn.Linear(width, width),
    nn.Tanh(),
    nn.Linear(width, 1)
).to(device)

loss_fn = lambda x, y: torch.sqrt(torch.mean(torch.pow(x - y, 2)))
# loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(mlp.parameters())

for i in tqdm(range(steps)):
    optimizer.zero_grad()
    x, y = batch_loader(batch_size)
    loss = loss_fn(mlp(x), y)
    if i % (steps // 20) == 0:
        print(loss.item())
    loss.backward()
    optimizer.step()
