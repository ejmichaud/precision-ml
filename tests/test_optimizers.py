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


