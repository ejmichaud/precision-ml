
import random
import time
from itertools import product, islice
from collections import defaultdict
from copy import deepcopy
from tqdm.auto import tqdm
import pickle

import numpy as np
from scipy import optimize

import pandas as pd
from sympy import parse_expr, lambdify

import torch
import torch.nn as nn

rmse_loss_fn_torch = lambda x, y: torch.sqrt(torch.mean(torch.pow(x-y, 2)))

if __name__ == '__main__':
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_default_dtype(torch.float64)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    
    eqn = 'I.11.19'
    equations = pd.read_csv("../equations.csv")
    row = equations[equations['Equation'] == eqn].iloc[0]
    dimension = int(row['# variables'])
    formula = row['Formula']
    variables = [row[f'v{i}_name'] for i in range(1, dimension+1)]
    ranges = [(row[f'v{i}_low'], row[f'v{i}_high']) for i in range(1, dimension+1)]
    target = lambdify(variables, parse_expr(formula))

    # create datasets
    ls = np.array([ranges[i][0] for i in range(dimension)])
    hs = np.array([ranges[i][1] for i in range(dimension)])
    x_i = np.random.uniform(low=ls, high=hs, size=(100000, dimension))
    y_i = target(*[x_i[:, i] for i in range(dimension)])

    cs = (hs + ls) / 2
    ws = (hs - ls) * 1.0
    ls, hs = cs - ws / 2, cs + ws / 2
    x_i_test = np.random.uniform(low=ls, high=hs, size=(10000, dimension))
    y_i_test = target(*[x_i_test[:, i] for i in range(dimension)])

    x_i = torch.from_numpy(x_i).to(device)
    y_i = torch.from_numpy(y_i).to(device).unsqueeze(dim=1)
    x_i_test = torch.from_numpy(x_i_test).to(device)
    y_i_test = torch.from_numpy(y_i_test).to(device).unsqueeze(dim=1)

    results = dict()

    for lr in tqdm([0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001]):
        results[lr] = dict()
        results[lr]['train'] = list()
        results[lr]['test'] = list()
        results[lr]['times'] = list()
    
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0) 
    
        width = 40
        mlp = nn.Sequential(
            nn.Linear(dimension, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 1)
        ).to(device)

        loss_fn = nn.MSELoss()
        optim = torch.optim.Adam(mlp.parameters(), lr=lr)

        results[lr]['start_time'] = time.time()
        for _ in tqdm(range(50000), leave=False):
            optim.zero_grad()
            y_pred = mlp(x_i)
            l = loss_fn(y_pred, y_i)
            with torch.no_grad():
                results[lr]['times'].append(time.time())
                results[lr]['train'].append(np.sqrt(l.item()))
                results[lr]['test'].append(rmse_loss_fn_torch(mlp(x_i_test), y_i_test).item())
            l.backward()
            optim.step()

    with open('../data/custom/compare-adam-lrs-eqn-2.pickle', 'wb') as handle:
        pickle.dump(results, handle)



