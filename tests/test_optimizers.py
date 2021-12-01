import pytest

import torch
import torch.nn as nn

from precisionml.optimizers import *

def test_conjugate_gradients0():
    params = nn.Parameter(torch.tensor([3., 5.]))
    def loss():
        return torch.sum(torch.tensor([1., 15.]) * torch.pow(params, 2))
    optimizer = ConjugateGradients([params])
    for _ in range(30):
        optimizer.zero_grad()
        l = loss()
        l.backward()
        optimizer.step(loss)
        if l.item() < 1e-30:
            break
    assert loss().item() < 1e-30

def test_conjugate_gradients1():
    params = nn.Parameter(torch.tensor([30, 100.]))
    def loss():
        return torch.sum(torch.tensor([80., 1.]) * torch.pow(params, 2))
    optimizer = ConjugateGradients([params])
    for _ in range(30):
        optimizer.zero_grad()
        l = loss()
        l.backward()
        optimizer.step(loss)
        if l.item() < 1e-30:
            break
        print(l.item())
    assert loss().item() < 1e-30

def test_conjugate_gradients2():
    params = nn.Parameter(torch.tensor([4., 4.]))
    def loss():
        return torch.sum(torch.tensor([500., 1.]) * torch.pow(params, 2))
    optimizer = ConjugateGradients([params])
    for _ in range(30):
        optimizer.zero_grad()
        l = loss()
        l.backward()
        optimizer.step(loss)
        if l.item() < 1e-30:
            break
        print(l.item())
    assert loss().item() < 1e-30

def test_conjugate_gradients3():
    params = nn.Parameter(torch.tensor([1., 1.]))
    def loss():
        return torch.sum(torch.tensor([5000., 1.]) * torch.pow(params, 2))
    optimizer = ConjugateGradients([params])
    for _ in range(30):
        optimizer.zero_grad()
        l = loss()
        l.backward()
        optimizer.step(loss)
        if l.item() < 1e-30:
            break
        print(l.item())
    assert loss().item() < 1e-30


def test_greedy_ensemble_optimizer0():
    params = nn.Parameter(torch.tensor([1., 1.])) 
    params2 = nn.Parameter(torch.tensor([1., 1.]))
    def get_loss(p):
        def loss():
            return torch.sum(torch.tensor([5000., 1.]) * torch.pow(p, 2))
        return loss
    opt1 = torch.optim.Adam([params])
    opt2 = ConjugateGradients([params])
    optimizer = GreedyEnsembleOptimizer([params], [opt1, opt2])

    optimizer2 = ConjugateGradients([params2])

    for _ in range(15):
        optimizer.zero_grad()
        l = get_loss(params)()
        l.backward()
        optimizer.step(get_loss(params))
        
        optimizer2.zero_grad()
        l = get_loss(params2)()
        l.backward()
        optimizer2.step(get_loss(params2))

        assert torch.all(params.data == params2.data)

        if l.item() < 1e-30:
            break

