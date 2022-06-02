#!/usr/bin/env python
# coding: utf-8
"""
The purpose of this script is to fit some of the AI Feynman datasets 
1st order linear simplex interpolation. Parses expression from .csv and
creates new random data points each run, instead of reading from the 
database.
"""

from collections import defaultdict
from itertools import islice, product
import random
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from scipy import spatial, interpolate 

import pandas as pd
from sympy import parse_expr, lambdify

import seml
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("feynman-simplex-subexperiment")
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
    spreadsheet = "/om2/user/ericjm/precision-ml/equations.csv"

    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

@ex.automain
def run(eqn, TRAIN_POINTS, N_TEST_POINTS, TEST_COMPACTIFICATION, spreadsheet, _log, seed):

    random.seed(seed)
    np.random.seed(seed)

    equations = pd.read_csv(spreadsheet)
    row = equations[equations['Equation'] == eqn].iloc[0]
    dimension = int(row['# variables'])
    formula = row['Formula']
    variables = [row[f'v{i}_name'] for i in range(1, dimension+1)]
    ranges = [(row[f'v{i}_low'], row[f'v{i}_high']) for i in range(1, dimension+1) ]
    target = lambdify(variables, parse_expr(formula))
    _log.debug(f"Fitting eqn {eqn}, of dimension {dimension}, given by {parse_expr(formula)}")

    ex.info['dimension'] = dimension

    _log.debug("Creating train and test set")
    
    # create datasets
    ls = np.array([ranges[i][0] for i in range(dimension)])
    hs = np.array([ranges[i][1] for i in range(dimension)])
    xs_train = np.random.uniform(low=ls, high=hs, size=(TRAIN_POINTS, dimension))
    ys_train = target(*[xs_train[:, i] for i in range(dimension)])

    cs = (hs + ls) / 2
    ws = (hs - ls) * TEST_COMPACTIFICATION
    ls, hs = cs - ws / 2, cs + ws / 2
    xs_test = np.random.uniform(low=ls, high=hs, size=(N_TEST_POINTS, dimension))
    ys_test = target(*[xs_test[:, i] for i in range(dimension)]) 

    if dimension == 1:
        xs_train = xs_train.reshape((TRAIN_POINTS,))
        ys_train = ys_train.reshape((TRAIN_POINTS,))
        xs_test = xs_test.reshape((N_TEST_POINTS,))
        ys_test = ys_test.reshape((N_TEST_POINTS,))
        
        _log.debug("'Training' model (creating triangulation, etc.)")
        f = interpolate.interp1d(xs_train, ys_train, kind='linear', bounds_error=False)

        _log.debug("Evaluating model") 
        preds = f(xs_test)
        valid_idx = ~np.isnan(preds)
        rmse_loss = np.sqrt(np.mean(np.power(preds[valid_idx] - ys_test[valid_idx], 2)))
        ex.info['test'] = rmse_loss.item() 
    else: 
        _log.debug("'Training' model (creating triangulation, etc.)")
        # train and evaluate 1st-order interpolation
        tri = spatial.Delaunay(xs_train)
        f = interpolate.LinearNDInterpolator(tri, ys_train)
        
        _log.debug("Evaluating model")
        preds = f(xs_test)
        valid_idx = ~np.isnan(preds)
        rmse_loss = np.sqrt(np.mean(np.power(preds[valid_idx] - ys_test[valid_idx], 2)))
        ex.info['test'] = rmse_loss.item()
    _log.debug("Test loss: {:.3e}".format(rmse_loss))
