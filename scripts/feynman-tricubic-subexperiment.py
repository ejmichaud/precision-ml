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

import pandas as pd
from sympy import parse_expr, lambdify

from ARBInterp.ARBInterp import tricubic

import seml
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("feynman-tricubic-subexperiment")
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
    eqn = "I.6.2b"
    TRAIN_POINTS = 50000
    N_TEST_POINTS = 30000
    TEST_COMPACTIFICATION = 0.8
    feynman_spreadsheet = "/om/user/ericjm/Downloads/FeynmanEquations.csv"
    
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

@ex.automain
def run(eqn, TRAIN_POINTS, N_TEST_POINTS, TEST_COMPACTIFICATION, feynman_spreadsheet, _log, seed):

    random.seed(seed)
    np.random.seed(seed)

    equations = pd.read_csv(feynman_spreadsheet)
    row = equations[equations['Filename'] == eqn].iloc[0]
    dimension = int(row['# variables'])
    assert dimension == 3
    formula = row['Formula']
    variables = [row[f'v{i}_name'] for i in range(1, 3+1)]
    ranges = [(row[f'v{i}_low'], row[f'v{i}_high']) for i in range(1, 3+1) ]
    target = lambdify(variables, parse_expr(formula))

    ex.info['dimension'] = dimension

    _log.debug("Creating train and test set")

    root_D = int(np.power(TRAIN_POINTS, 1/dimension))
    ex.info['TRAIN_POINTS_true'] = root_D ** dimension
    xis = [np.linspace(ranges[i][0], ranges[i][1], root_D) for i in range(dimension)]
    XIS = np.meshgrid(*xis)
    xs = np.array([X.flatten() for X in XIS]).T
    ys = target(*[xs[:, i] for i in range(dimension)])
    ys = np.expand_dims(ys, axis=1)
    field = np.concatenate([xs, ys], axis=1)

    f = tricubic(field)

    ls = np.array([ranges[i][0] for i in range(dimension)])
    hs = np.array([ranges[i][1] for i in range(dimension)])
    cs = (hs + ls) / 2
    ws = (hs - ls) * TEST_COMPACTIFICATION
    ls, hs = cs - ws / 2, cs + ws / 2

    xs_test = np.random.uniform(low=ls, high=hs, size=(N_TEST_POINTS, 3))
    ys_test = target(*[xs_test[:, i] for i in range(dimension)])
    preds, _ = f.Query(xs_test)
    preds = preds.squeeze()
    valid_idx = ~np.isnan(preds)
    rmse_loss = np.sqrt(np.mean(np.power(preds[valid_idx] - ys_test[valid_idx], 2))) 
    ex.info['test'] = rmse_loss.item()

