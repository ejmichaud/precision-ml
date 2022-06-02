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

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("feynman-interpolation-experiments")
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def cfg():
    MAX_D = 4
    REPEATS = 3
    TRAIN_POINTS = [4**n for n in range(3, 10)]
    N_TEST_POINTS = 10000
    TEST_COMPACTIFICATION = 0.9
    data_dir = "/home/eric/Downloads/Feynman_with_units"

@ex.automain
def run(REPEATS, TRAIN_POINTS, N_TEST_POINTS, TEST_COMPACTIFICATION, MAX_D, data_dir):

    for i, file in tqdm(list(enumerate(Path(data_dir).iterdir()))):
        data = np.loadtxt(file, dtype=np.float64)
        _, size = data.shape
        dimension = size - 1
        if 2 <= dimension <= MAX_D:

            xs = data[:, :dimension]
            ys = data[:, dimension:]
            ls = xs.min(axis=0)
            hs = xs.max(axis=0)
            cs = (hs + ls) / 2
            ws = (hs - ls) / 2
            ls, hs = cs - ws / 2, cs + ws / 2
            interior_points = [k for k in range(len(xs)) if 
                        np.all((xs[k] > TEST_COMPACTIFICATION * ls) & (xs[k] < TEST_COMPACTIFICATION * hs))]
            # choose test data
            test_subset = np.random.choice(interior_points, size=N_TEST_POINTS)
            xs_test = xs[test_subset, :]
            ys_test = ys[test_subset, :]

            eqn = file.parts[-1]
            ex.info[eqn] = dict()
            ex.info[eqn]['dimension'] = dimension
            ex.info[eqn]['zeroth'] = defaultdict(list)
            ex.info[eqn]['first'] = defaultdict(list)

            for _, D in tqdm(list(product(range(REPEATS), TRAIN_POINTS)), leave=False):

                # choose training data
                subset = np.random.choice([n for n in range(len(xs)) if n not in test_subset], size=D)
                xs_train = xs[subset, :]
                ys_train = ys[subset, :]
                tri = spatial.Delaunay(xs_train)

                # train and evaluate 0th-order interpolation
                # f = interpolate.NearestNDInterpolator(tri, ys_subset)
                # preds = f(xs_test)
                # valid_idx = ~np.isnan(preds)
                # rmse_loss = np.sqrt(np.mean(np.power(preds[valid_idx] - ys_test[valid_idx], 2)))
                # ex.info[eqn]['zeroth'][D].append(rmse_loss.item())

                # train and evaluate 1st-order interpolation
                f = interpolate.LinearNDInterpolator(tri, ys_train)
                preds = f(xs_test)
                valid_idx = ~np.isnan(preds)
                rmse_loss = np.sqrt(np.mean(np.power(preds[valid_idx] - ys_test[valid_idx], 2)))
                ex.info[eqn]['first'][D].append(rmse_loss.item())




