#!/usr/bin/env python
# coding: utf-8
"""
The purpose of this script is to test the efficacy of boosting
on teacher-student setups of varying input dimension. We know that
boosting can deliver significant gains in 1 dimension. But what
about higher dimensions?
"""

from itertools import islice, product
import random
from re import I
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

from scipy import optimize, special

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("feynman-boosting-experiments")
ex.captured_out_filter = apply_backspaces_and_linefeeds

rmse_loss_fn_torch = lambda x, y: torch.sqrt(torch.mean(torch.pow(x-y, 2)))
qmqe_loss_fn_torch = lambda x, y: torch.pow(torch.mean(torch.pow(x-y, 4)), 1/4)
smse_loss_fn_torch = lambda x, y: torch.pow(torch.mean(torch.pow(x-y, 6)), 1/6)
mse_loss_fn_torch = nn.MSELoss()

def lp_norm(p):
    def loss(x, y):
        return torch.pow(torch.mean(torch.pow(torch.abs(x-y), p)), 1/p)
    return loss

def dl_loss(epsilon):
    def loss(x, y):
        return torch.mean(0.5 * torch.log2(1 + torch.pow((x-y) / epsilon, 2)))
    return loss

def loss(param_vector, lenghts, shapes, 
             mlp, loss_fn, x, y, device):
    l = 0
    for i, param in enumerate(mlp.parameters()):
        param_data = param_vector[l:l+lenghts[i]]
        l += lenghts[i]
        param_data_shaped = param_data.reshape(shapes[i])
        param.data = torch.tensor(param_data_shaped).to(device)
    return loss_fn(mlp(x.to(device)), y).detach().cpu().numpy()

def gradient(param_vector, lenghts, shapes, 
             mlp, loss_fn, x, y, device):
    l = 0
    for i, param in enumerate(mlp.parameters()):
        param_data = param_vector[l:l+lenghts[i]]
        l += lenghts[i]
        param_data_shaped = param_data.reshape(shapes[i])
        param.data = torch.tensor(param_data_shaped).to(device)
    loss_fn(mlp(x.to(device)), y).backward()
    grads = []
    for param in mlp.parameters():
        grads.append(param.grad.detach().clone().cpu().numpy().flatten())
        param.grad = None
    mlp.zero_grad()
    return np.concatenate(grads)


@ex.config
def cfg():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    width = 30
    depth = 3
    activation = nn.Tanh
    dimensions = list(range(1, 10))


def train_and_observe(i, width, depth, activation, dimension, device, dtype, seed):
    
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed + i)
    random.seed(seed + i)
    np.random.seed(seed + i)

    ex.info[dimension] = dict()

    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(dimension, 3))
            layers.append(activation())
        elif i == depth - 1:
            layers.append(nn.Linear(3, 1))
        else:
            layers.append(nn.Linear(3, 3))
            layers.append(activation())
    teacher = nn.Sequential(*layers).to(device)

    x_i = np.random.uniform(low=-1, high=1, size=(100000, dimension))
    x_i = torch.from_numpy(x_i).to(device)
    with torch.no_grad():
        y_i = teacher(x_i)
    
    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(dimension, width))
            layers.append(activation())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 1))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation())
    mlp = nn.Sequential(*layers).to(device)

    params = []
    shapes = []
    lengths = []
    for param in mlp.parameters():
        param_np = param.data.detach().clone().cpu().numpy()
        shapes.append(param_np.shape)
        param_np_flat = param_np.flatten()
        lengths.append(len(param_np_flat))
        params.append(param_np_flat)

    param_vector = np.concatenate(params)

    result = optimize.minimize(loss,
                        param_vector, 
                        args=(lengths, shapes, mlp, mse_loss_fn_torch, x_i, y_i, device),
                        jac=gradient,
                        method='BFGS',
                        options={
                            'disp': False,
                            'gtol': 1e-40,
                            'maxiter': 5000,
                        })

    l = 0
    for j, param in enumerate(mlp.parameters()):
        param_data = result.x[l:l+lengths[j]]
        l += lengths[j]
        param_data_shaped = param_data.reshape(shapes[j])
        param.data = torch.tensor(param_data_shaped).to(device)

    with torch.no_grad():
        ex.info[dimension]['whole'] = mse_loss_fn_torch(mlp(x_i), y_i).item()

    f1 = nn.Sequential(
        nn.Linear(dimension, int(width/2)),
        nn.Tanh(),
        nn.Linear(int(width/2), int(width/2)),
        nn.Tanh(),
        nn.Linear(int(width/2), 1)
    ).to(device)

    params1 = []
    shapes1 = []
    lengths1 = []
    for param in f1.parameters():
        param_np = param.data.detach().clone().cpu().numpy()
        shapes1.append(param_np.shape)
        param_np_flat = param_np.flatten()
        lengths1.append(len(param_np_flat))
        params1.append(param_np_flat)

    param_vector1 = np.concatenate(params1)

    result = optimize.minimize(loss,
                        param_vector1, 
                        args=(lengths1, shapes1, f1, mse_loss_fn_torch, x_i, y_i, device),
                        jac=gradient,
                        method='BFGS',
                        options={
                            'disp': False,
                            'gtol': 1e-40,
                            'maxiter': 5000,
                        })

    l = 0
    for j, param in enumerate(f1.parameters()):
        param_data = result.x[l:l+lengths1[j]]
        l += lengths1[j]
        param_data_shaped = param_data.reshape(shapes1[j])
        param.data = torch.tensor(param_data_shaped).to(device)
        

    with torch.no_grad():
        c1 = 1 / torch.mean(torch.abs(y_i - f1(x_i))).item()
        yd1_i = c1 * (y_i - f1(x_i))


    f2 = nn.Sequential(
        nn.Linear(dimension, int(width/2)),
        nn.Tanh(),
        nn.Linear(int(width/2), int(width/2)),
        nn.Tanh(),
        nn.Linear(int(width/2), 1)
    ).to(device)

    params2 = []
    shapes2 = []
    lengths2 = []
    for param in f2.parameters():
        param_np = param.data.detach().clone().cpu().numpy()
        shapes2.append(param_np.shape)
        param_np_flat = param_np.flatten()
        lengths2.append(len(param_np_flat))
        params2.append(param_np_flat)

    param_vector2 = np.concatenate(params2)


    result = optimize.minimize(loss,
                        param_vector2, 
                        args=(lengths2, shapes2, f2, mse_loss_fn_torch, x_i, yd1_i, device),
                        jac=gradient,
                        method='BFGS',
                        options={
                            'disp': False,
                            'gtol': 1e-40,
                            'maxiter': 5000,
                        })

    l = 0
    for j, param in enumerate(f2.parameters()):
        param_data = result.x[l:l+lengths2[j]]
        l += lengths2[j]
        param_data_shaped = param_data.reshape(shapes2[j])
        param.data = torch.tensor(param_data_shaped).to(device)

    with torch.no_grad():
        ex.info[dimension]['primary'] = mse_loss_fn_torch(f1(x_i), y_i).item()
        ex.info[dimension]['boosted'] = mse_loss_fn_torch(f1(x_i) + (1/c1)*f2(x_i), y_i).item()


@ex.automain
def run(width, depth, activation, dimensions, device, dtype, seed):

    for i, dimension in tqdm(list(enumerate(dimensions))):
        train_and_observe(i, width, depth, activation, dimension, device, dtype, seed)

