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

from scipy import spatial, interpolate 

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
    eqn = 'I.10.7'
    TRAIN_POINTS = 50000
    N_TEST_POINTS = 30000
    TEST_COMPACTIFICATION = 0.8
    data_dir = "/home/eric/Downloads/Feynman_with_units"

    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

@ex.automain
def run(eqn, TRAIN_POINTS, N_TEST_POINTS, TEST_COMPACTIFICATION, data_dir, _log, seed):

    random.seed(seed)
    np.random.seed(seed)

    file = Path(data_dir) / eqn
    data = np.loadtxt(file, dtype=np.float64)
    _, size = data.shape
    dimension = size - 1
    _log.debug(f"Dimension of problem {eqn} is {dimension}")

    ex.info['dimension'] = dimension

    _log.debug("Creating train and test set")
    xs = data[:, :dimension]
    ys = data[:, dimension:]
    ls = xs.min(axis=0)
    hs = xs.max(axis=0)
    cs = (hs + ls) / 2
    ws = (hs - ls) * TEST_COMPACTIFICATION
    ls, hs = cs - ws / 2, cs + ws / 2
    interior_points = [k for k in range(len(xs)) if 
                np.all((xs[k] > ls) & (xs[k] < hs))]
    # choose test data
    test_subset = np.random.choice(interior_points, size=N_TEST_POINTS)
    xs_test = xs[test_subset, :]
    ys_test = ys[test_subset, :]

    # choose training data
    subset = np.random.choice([n for n in range(len(xs)) if n not in test_subset], size=TRAIN_POINTS)
    xs_train = xs[subset, :]
    ys_train = ys[subset, :]
    tri = spatial.Delaunay(xs_train)

    _log.debug("'Training' model (creating triangulation, etc.)")
    # train and evaluate 1st-order interpolation
    f = interpolate.LinearNDInterpolator(tri, ys_train)
    _log.debug("Evaluating model")
    preds = f(xs_test)
    valid_idx = ~np.isnan(preds)
    rmse_loss = np.sqrt(np.mean(np.power(preds[valid_idx] - ys_test[valid_idx], 2)))
    ex.info['test'] = rmse_loss.item()


