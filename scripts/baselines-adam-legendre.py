#!/usr/bin/env python
# coding: utf-8
"""
The purpose of this script is to provide baseline results of how SGD
performs on fitting Legendre polynomials R^1 -> R^1.
"""

from itertools import islice, product
import random
from re import I
import time

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

from scipy import optimize, special

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("baselines-adam-legendre")
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


@ex.config
def cfg():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    # widths = [10, 20, 30]
    widths = [24]
    # depths = [2, 3, 4]
    depths = [3, 4]
    # activations = [nn.Sigmoid, nn.Tanh, nn.Softplus]
    activations = [nn.Tanh]
    trials = 8 # the number of times to test each architecture
    # lrs = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
    lrs = [1e-1, 1e-3, 1e-5, 1e-7]
    steps = 25000
    # orders = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    orders = [0, 1, 2, 3, 4, 5]

def train_and_observe(i, width, depth, activation, trial, lr, steps, order, device, dtype, seed):
    
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed + i)
    random.seed(seed + i)
    np.random.seed(seed + i)
    
    key = f"{width}::{depth}::{activation}::{trial}::{lr}::{order}"

    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(1, width))
            layers.append(activation())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 1))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation())
    mlp = nn.Sequential(*layers).to(device)

    x_n = np.random.uniform(low=-1, high=1, size=(100000, 1))
    y_n = special.legendre(order)(x_n)
    x_n = torch.from_numpy(x_n).to(device)
    y_n = torch.from_numpy(y_n).to(device)

    optim = torch.optim.Adam(mlp.parameters(), lr=lr)

    results = dict()
    results['steps'] = []
    results['timestamps'] = []
    results['losses'] = []
    for step in range(steps):
        optim.zero_grad()
        l = mse_loss_fn_torch(mlp(x_n), y_n)
        l.backward()
        optim.step()
        results['steps'].append(step)
        results['timestamps'].append(time.time())
        results['losses'].append(l.item())
    
    ex.info[key] = results


@ex.automain
def run(widths, depths, activations, trials, lrs, steps, orders, device, dtype, seed):

    for i, (width, depth, activation, trial, lr, order) in tqdm(list(enumerate(product(widths, depths, activations, range(trials), lrs, orders)))):
        train_and_observe(i, width, depth, activation, trial, lr, steps, order, device, dtype, seed)

