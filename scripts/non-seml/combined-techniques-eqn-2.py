
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

def get_func(model: nn.Module, loss_fn, x, y):
    def func(*p_tensors):
        assert len(list(model.parameters())) == len(p_tensors)
        assert type(model) is nn.Sequential
        i = 0
        z = x
        for module in model:
            if type(module) is nn.Linear:
                if module.bias is None:
                    z = nn.functional.linear(z, p_tensors[i], None)
                    i += 1
                else:
                    z = nn.functional.linear(z, p_tensors[i], p_tensors[i+1])
                    i += 2
            if type(module) in [nn.Tanh, nn.Sigmoid, nn.ReLU]:
                z = module(z)
        return loss_fn(z, y)
    return func

def hessian_autograd(model, loss_fn, x, y):
    H = torch.autograd.functional.hessian(get_func(model, loss_fn, x, y), tuple(model.parameters()))
    return H

def hvp_autograd(model, v, loss_fn, x, y):
    _, Hv = torch.autograd.functional.hvp(get_func(model, loss_fn, x, y), tuple(model.parameters()), v)
    return Hv

def decompose_hessian(hessian, lengths, shapes):
    H_matrix = torch.zeros(sum(lengths), sum(lengths))
    for i, j in product(range(len(shapes)), range(len(shapes))):
        i0, i1 = sum(lengths[:i+1]) - lengths[i], sum(lengths[:i+1])
        j0, j1 = sum(lengths[:j+1]) - lengths[j], sum(lengths[:j+1])
        H_matrix[i0:i1, j0:j1] = hessian[i][j].reshape(lengths[i], lengths[j])
    vals, vecs = torch.linalg.eigh(H_matrix)
    vs = []
    for i in range(sum(lengths)):
        vec = vecs[:, i]
        v = tuple(vec[sum(lengths[:i]):sum(lengths[:i+1])].reshape(shapes[i]) for i in range(len(lengths)))
        vs.append(v)
    return vals, vs

def dot_prod(u, v):
    assert len(u) == len(v)
    result = 0
    for i in range(len(u)):
        result += torch.sum(u[i] * v[i])
    return result.item()

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


def loss(param_vector, lengths, shapes, 
             mlp, loss_fn, x, y, device):
    l = 0
    for i, param in enumerate(mlp.parameters()):
        param_data = param_vector[l:l+lengths[i]]
        l += lengths[i]
        param_data_shaped = param_data.reshape(shapes[i])
        param.data = torch.tensor(param_data_shaped).to(device)
    return loss_fn(mlp(x.to(device)), y).detach().cpu().numpy()

def gradient(param_vector, lengths, shapes, 
             mlp, loss_fn, x, y, device):
    l = 0
    for i, param in enumerate(mlp.parameters()):
        param_data = param_vector[l:l+lengths[i]]
        l += lengths[i]
        param_data_shaped = param_data.reshape(shapes[i])
        param.data = torch.tensor(param_data_shaped).to(device)
    loss_fn(mlp(x.to(device)), y).backward()
    grads = []
    for param in mlp.parameters():
        grads.append(param.grad.detach().clone().cpu().numpy().flatten())
        param.grad = None
    mlp.zero_grad()
    return np.concatenate(grads)

def logarithmic_line_search(f, low=1e-12, high=1e12, points=15, depth=4):
    """Performs line search of positive parameter between 
    `low` and `high`, minimizing objective function `f`.
    Only searches through positive values of parameter,
    so define your objective function `f` accordingly."""
    assert low > 0 and high > 0
    best_loss = float('inf')
    best_alpha = None
    for d in range(depth):
        alphas = np.exp(np.linspace(np.log(low), np.log(high), points))
        losses = [f(alpha) for alpha in alphas]
        i_min = np.argmin(losses)
        if losses[i_min] < best_loss:
            best_alpha = alphas[i_min]
        else:
            return alphas[i_min]
        if i_min == 0:
            high = alphas[i_min + 1]
            low = alphas[i_min] * (alphas[i_min] / alphas[i_min + 1])
        elif i_min == points - 1:
            low = alphas[i_min - 1]
            high = alphas[i_min] * (alphas[i_min] / alphas[i_min - 1])
        else:
            low = alphas[i_min - 1]
            high = alphas[i_min + 1]
    return best_alpha

if __name__ == '__main__':
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.set_default_dtype(torch.float64)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    eqn = 'I.11.19'
    equations = pd.read_csv("../../equations.csv")
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
    
    width = 20
    mlp = nn.Sequential(
        nn.Linear(dimension, width),
        nn.Tanh(),
        nn.Linear(width, width),
        nn.Tanh(),
        nn.Linear(width, 1)
    ).to(device)

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

    results['train1'] = list()
    results['test1'] = list()
    results['times1'] = list()
    def callback(xk):
        results['times1'].append(time.time())
        results['train1'].append(loss(xk, lengths, shapes, mlp, rmse_loss_fn_torch, x_i, y_i, device).item())
        results['test1'].append(loss(xk, lengths, shapes, mlp, rmse_loss_fn_torch, x_i_test, y_i_test, device).item())

    results['start_time1'] = time.time()
    result = optimize.minimize(loss,
                        param_vector, 
                        args=(lengths, shapes, mlp, mse_loss_fn_torch, x_i, y_i, device),
                        jac=gradient,
                        method='BFGS',
                        options={
                            'disp': True,
                            'gtol': 1e-60,
                            'maxiter': 25000,
            # #                                'finite_diff_rel_step': 1e-15
                        },
                        callback=callback
                    ) 
    mlp.zero_grad()
    solution_params = [p.detach().clone() for p in mlp.parameters()]

    for j, p in enumerate(mlp.parameters()):
        p.data = solution_params[j].clone()

    curvature_threshold = 1e-15
    results['train2'] = list()
    results['test2'] = list()
    results['times2'] = list()
    results['start_time2'] = time.time()
    for _ in tqdm(range(500)):
        # parameters we start the step at
        step_params = [p.detach().clone() for p in mlp.parameters()]

        # compute gradient and hessian
        l = mse_loss_fn_torch(mlp(x_i), y_i)
        l.backward()
        g = [p.grad.detach().clone().cpu() for p in mlp.parameters()]
        mlp.zero_grad()
        hessian = hessian_autograd(mlp, mse_loss_fn_torch, x_i, y_i)
        vals, vs = decompose_hessian(hessian, lengths, shapes)
        mlp.zero_grad()

        # compute projected gradient
        v_g = []
        with torch.no_grad():
            for i, p in enumerate(mlp.parameters()):
                g_i_proj = torch.zeros(p.shape).to(device)
                for j, (val, v) in enumerate(zip(vals.numpy(), vs)):
                    if np.abs(val) < curvature_threshold:
                        g_i_proj += (v[i] * dot_prod(v, g)).to(device)
                v_g.append(g_i_proj)

        # helper function for computing loss along direction v_g
        @torch.no_grad()
        def f(alpha):
            for j, p in enumerate(mlp.parameters()):
                p.data = step_params[j].clone() - alpha * v_g[j]
            l = mse_loss_fn_torch(mlp(x_i), y_i).item()
            for j, p in enumerate(mlp.parameters()):
                p.data = step_params[j].clone()
            return l

        # perform search and step
        lr = logarithmic_line_search(f, low=1e-30, points=70)
        with torch.no_grad():
            for i, p in enumerate(mlp.parameters()):
                p.add_(-lr * v_g[i])

        results['times2'].append(time.time())
        results['train2'].append(rmse_loss_fn_torch(mlp(x_i), y_i).item())
        results['test2'].append(rmse_loss_fn_torch(mlp(x_i_test), y_i_test).item())

    solution_params2 = [p.detach().clone() for p in mlp.parameters()]
    
    with torch.no_grad():
        c = 1 / torch.mean(torch.abs(y_i - mlp(x_i))).item()
        yd_i = c * (y_i - mlp(x_i))

    width = 20
    mlp2 = nn.Sequential(
        nn.Linear(dimension, width),
        nn.Tanh(),
        nn.Linear(width, width),
        nn.Tanh(),
        nn.Linear(width, 1)
    ).to(device)

    params2 = []
    shapes2 = []
    lengths2 = []
    for param in mlp2.parameters():
        param_np = param.data.detach().clone().cpu().numpy()
        shapes2.append(param_np.shape)
        param_np_flat = param_np.flatten()
        lengths2.append(len(param_np_flat))
        params2.append(param_np_flat)
    param_vector2 = np.concatenate(params2)

    results['train3'] = list()
    results['test3'] = list()
    results['times3'] = list()
    def callback(xk):
        """xk is the param vector for mlp2"""
        l = 0
        for i, param in enumerate(mlp2.parameters()):
            param_data = xk[l:l+lengths2[i]]
            l += lengths2[i]
            param_data_shaped = param_data.reshape(shapes2[i])
            param.data = torch.tensor(param_data_shaped).to(device)
        results['times3'].append(time.time())
        results['train3'].append(rmse_loss_fn_torch(mlp(x_i) + (1/c)*mlp2(x_i), y_i).item())
        results['test3'].append(rmse_loss_fn_torch(mlp(x_i_test) + (1/c)*mlp2(x_i_test), y_i_test).item())

    results['start_time3'] = time.time()
    result = optimize.minimize(loss,
                        param_vector2, 
                        args=(lengths2, shapes2, mlp2, mse_loss_fn_torch, x_i, yd_i, device),
                        jac=gradient,
                        method='BFGS',
                        options={
                            'disp': True,
                            'gtol': 1e-60,
                            'maxiter': 25000,
    #                                'finite_diff_rel_step': 1e-15
                        },
                        callback=callback
                        )
    mlp2.zero_grad()

    boost_solution_params = [p.detach().clone() for p in mlp2.parameters()]

    for j, p in enumerate(mlp2.parameters()):
        p.data = boost_solution_params[j].clone()

    curvature_threshold = 1e-15
    results['train4'] = list()
    results['test4'] = list()
    results['times4'] = list()
    results['start_time4'] = time.time()
    for _ in tqdm(range(500)):
        # parameters we start the step at
        step_params = [p.detach().clone() for p in mlp2.parameters()]

        # compute gradient and hessian
        l = mse_loss_fn_torch(mlp2(x_i), yd_i)
        l.backward()
        g = [p.grad.detach().clone().cpu() for p in mlp2.parameters()]
        mlp2.zero_grad()
        hessian = hessian_autograd(mlp2, mse_loss_fn_torch, x_i, yd_i)
        vals, vs = decompose_hessian(hessian, lengths, shapes)
        mlp2.zero_grad()

        # compute projected gradient
        v_g = []
        with torch.no_grad():
            for i, p in enumerate(mlp2.parameters()):
                g_i_proj = torch.zeros(p.shape).to(device)
                for j, (val, v) in enumerate(zip(vals.numpy(), vs)):
                    if np.abs(val) < curvature_threshold:
                        g_i_proj += (v[i] * dot_prod(v, g)).to(device)
                v_g.append(g_i_proj)

        # helper function for computing loss along direction v_g
        @torch.no_grad()
        def f(alpha):
            for j, p in enumerate(mlp2.parameters()):
                p.data = step_params[j].clone() - alpha * v_g[j]
            l = mse_loss_fn_torch(mlp2(x_i), yd_i).item()
            for j, p in enumerate(mlp2.parameters()):
                p.data = step_params[j].clone()
            return l

        # perform search and step
        lr = logarithmic_line_search(f, low=1e-50, points=70)
        with torch.no_grad():
            for i, p in enumerate(mlp2.parameters()):
                p.add_(-lr * v_g[i])

        results['times4'].append(time.time())
        results['train4'].append(rmse_loss_fn_torch(mlp(x_i) + (1/c)*mlp2(x_i), y_i).item())
        results['test4'].append(rmse_loss_fn_torch(mlp(x_i_test) + (1/c)*mlp2(x_i_test), y_i_test).item())

    with open('../../data/custom/combined-techniques-eqn-2.pickle', 'wb') as handle:
        pickle.dump(results, handle)



