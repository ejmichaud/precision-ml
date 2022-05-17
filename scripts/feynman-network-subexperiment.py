#!/usr/bin/env python
# coding: utf-8
"""
The purpose of this script is to fit some of the AI Feynman datasets using
0th and 1st order interpolation. We only fit problems below a certain dimension
since computing the simplices gets hard in more dimensions!
"""

from collections import defaultdict
from itertools import islice, product
import random
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn

import seml
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("feynman-network-subexperiment")
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

# --------------------------
#    ,-------------.
#   (_\  CONFIG     \
#      |    OF      |
#      |    THE     |
#     _| EXPERIMENT |
#    (_/_____(*)___/
#             \\
#              ))
#              ^
# --------------------------
@ex.config
def cfg():
    eqn = 'I.10.7'
    width = 100
    depth = 3
    lr = 1e-3
    activation = nn.ReLU
    N_TEST_POINTS = 30000
    TEST_COMPACTIFICATION = 0.9
    DATA_RATIO = 1.0
    MAX_TRAIN_ITERS = 25000
    MAX_BATCH_SIZE = 30000
    data_dir = "/home/eric/Downloads/Feynman_with_units"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------
@ex.automain
def run(eqn, width, depth, lr, activation, 
            N_TEST_POINTS, TEST_COMPACTIFICATION, DATA_RATIO, MAX_TRAIN_ITERS, MAX_BATCH_SIZE,
            data_dir, device, dtype, seed, _log):
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    file = Path(data_dir) / eqn
    data = np.loadtxt(file, dtype=np.float64)
    _, size = data.shape
    dimension = size - 1
    _log.debug(f"Dimension of problem {eqn} is {dimension}")
    
    ex.info['dimension'] = dimension
    ex.info['train'] = list()
    ex.info['test'] = list()
    
    xs = data[:, :dimension]
    ys = data[:, dimension:]
    ls = xs.min(axis=0)
    hs = xs.max(axis=0)
    cs = (hs + ls) / 2
    ws = (hs - ls) / 2
    ls, hs = cs - ws / 2, cs + ws / 2
    interior_points = [k for k in range(len(xs)) if 
                np.all((xs[k] > TEST_COMPACTIFICATION * ls) & (xs[k] < TEST_COMPACTIFICATION * hs))]
    xs = torch.from_numpy(xs).to(device)
    ys = torch.from_numpy(ys).to(device)
    # choose test data
    test_subset = np.random.choice(interior_points, size=N_TEST_POINTS)
    xs_test = xs[test_subset, :]
    ys_test = ys[test_subset, :]
    # create model
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
    _log.debug("Created model.")

    D = sum(t.numel() for t in mlp.parameters())
    assert D == ((dimension * width + width) + (depth - 2)*(width * width + width) + (width + 1))
    # adjust to match on data ratio
    D = int(DATA_RATIO * D)

    # choose training data
    subset = np.random.choice([n for n in range(len(xs)) if n not in test_subset], size=D)
    xs_train = xs[subset, :]
    ys_train = ys[subset, :]
    _log.debug("Chose train set.")

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(mlp.parameters(), lr=lr)

    min_train = float('inf')
    min_test = float('inf')

    k = 0
    for step in range(MAX_TRAIN_ITERS):
        optim.zero_grad()
        if D <= MAX_BATCH_SIZE:
            ys_pred = mlp(xs_train)
            l = loss_fn(ys_train, ys_pred)
        else:
            sample_idx = torch.arange(k, k+MAX_BATCH_SIZE, 1) % D
            xs_batch, ys_batch = xs_train[sample_idx], ys_train[sample_idx]
            ys_pred = mlp(xs_batch)
            l = loss_fn(ys_batch, ys_pred)
            k += MAX_BATCH_SIZE
        l.backward()
        optim.step()
        with torch.no_grad():
            train_l = torch.sqrt(l).item()
            test_l = torch.sqrt(torch.mean(torch.pow(mlp(xs_test) - ys_test, 2))).item()
            min_train = train_l if train_l < min_train else min_train
            min_test = test_l if test_l < min_test else min_test
            if step % 100 == 0:
                ex.info['train'].append(train_l)
                ex.info['test'].append(test_l)
        if step % (MAX_TRAIN_ITERS // 10) == 0:
            _log.debug("{:.0f}% done with training".format(step / MAX_TRAIN_ITERS * 100))
    ex.info['min_train'] = min_train
    ex.info['min_test'] = min_test
