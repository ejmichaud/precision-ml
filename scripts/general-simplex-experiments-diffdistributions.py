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
ex = Experiment("general-simplex-experiments-diffdistributions")
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def cfg():
    REPEATS = 5
    TRAIN_POINTS = [4**n for n in range(3, 10)]
    N_TEST_POINTS = 30000
    TEST_COMPACTIFICATION = 0.5
    TARGETS = {
        # 'x^2': {
        #     'dimension': 1,
        #     'range': ((-1, 1),),
        #     'function': lambda x: np.power(x, 2)
        # },
        # 'sqrt(x)': {
        #     'dimension': 1,
        #     'range': ((0, 1),),
        #     'function': lambda x: np.sqrt(x)
        # },
        # 'cos(1000*x)': {
        #     'dimension': 1,
        #     'range': ((-1, 1),),
        #     'function': lambda x:  np.cos(1000*x)
        # },
        'x*y': {
            'dimension': 2,
            'range': ((-1, 1), (-1, 1)),
            'function': lambda x, y: x*y
        },
        'x^2 * y^2': {
            'dimension': 2,
            'range': ((-2, 2), (-2, 2)),
            'function': lambda x, y: np.power(x, 2) * np.power(y, 2)
        }, 
        'cos(10*x*y)': {
            'dimension': 2,
            'range': ((-1, 1), (-1, 1)),
            'function': lambda x, y: np.cos(10*x*y)
        },
        'x*y*z': {
            'dimension': 3,
            'range': ((-1, 1), (-1, 1), (-1, 1)),
            'function': lambda x, y, z: x*y*z
        },
        'x * y^2 * z^3': {
            'dimension': 3,
            'range': ((-1, 1), (-1, 1), (-1, 1)),
            'function': lambda x, y, z: x * np.power(y, 2) * np.power(z, 3)
        },
        'cos(10*x*y - 5*z)': {
            'dimension': 3,
            'range': ((-1, 1), (-1, 1), (-1, 1)),
            'function': lambda x, y, z: np.cos(10*x*y - 5*z)
        }
    }


@ex.automain
def run(REPEATS, TRAIN_POINTS, N_TEST_POINTS, TEST_COMPACTIFICATION, TARGETS):

    for eqn in tqdm(TARGETS):

        dimension = TARGETS[eqn]['dimension']
        ranges = TARGETS[eqn]['range']
        target_fn = TARGETS[eqn]['function']
        
        ex.info[eqn] = dict()
        ex.info[eqn]['regular'] = defaultdict(list)
        ex.info[eqn]['random'] = defaultdict(list)

        for D in tqdm(TRAIN_POINTS, leave=False):
            D = int(D)
            if dimension == 1:
                # Regular grid first
                xs = np.linspace(ranges[0][0], ranges[0][1], D)
                ys = target_fn(xs)
                fit = interpolate.interp1d(xs, ys, kind='linear', fill_value=np.nan, bounds_error=False)
                for _ in tqdm(range(REPEATS), leave=False):
                    l, h = ranges[0][0], ranges[0][1]
                    c, w = (h + l) / 2, (h - l) * TEST_COMPACTIFICATION
                    l, h = c - w / 2, c + w / 2
                    xs_test = np.random.uniform(low=l, high=h, size=(N_TEST_POINTS,))
                    ys_test = target_fn(xs_test)
                    preds = fit(xs_test)
                    valid_idx = ~np.isnan(preds)
                    rmse_loss = np.sqrt(np.mean(np.power(preds[valid_idx] - ys_test[valid_idx], 2))).item()
                    ex.info[eqn]['regular'][D].append(rmse_loss)

                # Random points 
                for _ in tqdm(range(REPEATS), leave=False):
                    xs = np.random.uniform(low=ranges[0][0], high=ranges[0][1], size=(D,))
                    ys = target_fn(xs)
                    fit = interpolate.interp1d(xs, ys, kind='linear', fill_value=np.nan, bounds_error=False)
                    l, h = ranges[0][0], ranges[0][1]
                    c, w = (h + l) / 2, (h - l) * TEST_COMPACTIFICATION
                    l, h = c - w / 2, c + w / 2
                    xs_test = np.random.uniform(low=l, high=h, size=(N_TEST_POINTS,))
                    ys_test = target_fn(xs_test)
                    preds = fit(xs_test)
                    valid_idx = ~np.isnan(preds)
                    rmse_loss = np.sqrt(np.mean(np.power(preds[valid_idx] - ys_test[valid_idx], 2))).item()
                    ex.info[eqn]['random'][D].append(rmse_loss) 
            else:
                # Do regular grid first
                root_D = int(np.power(D, 1/dimension))
                xis = [np.linspace(ranges[i][0], ranges[i][1], root_D) for i in range(dimension)]
                XIS = np.meshgrid(*xis)
                xs = np.array([X.flatten() for X in XIS]).T
                ys = target_fn(*[xs[:, i] for i in range(dimension)])

                tri = spatial.Delaunay(xs)
                fit = interpolate.LinearNDInterpolator(tri, ys)

                for _ in tqdm(range(REPEATS), leave=False):
                    xis_test = []
                    for i in range(dimension):
                        l, h = ranges[i][0], ranges[i][1]
                        c, w = (h + l) / 2, (h - l) * TEST_COMPACTIFICATION
                        l, h = c - w / 2, c + w / 2
                        xis_test.append(np.random.uniform(low=l, high=h, size=(N_TEST_POINTS,)))
                    xs_test = np.array(xis_test).T
                    ys_test = target_fn(*[xis_test[i] for i in range(dimension)])
                    
                    preds = fit(xs_test)
                    valid_idx = ~np.isnan(preds)
                    rmse_loss = np.sqrt(np.mean(np.power(preds[valid_idx] - ys_test[valid_idx], 2))).item()
                    ex.info[eqn]['regular'][D].append(rmse_loss)

                # Do random points now
                for _ in tqdm(range(REPEATS), leave=False):

                    xis = [np.random.uniform(low=ranges[i][0], high=ranges[i][1], size=(D,)) for i in range(dimension)] 
                    xs = np.array(xis).T 
                    ys = target_fn(*[xs[:, i] for i in range(dimension)])

                    tri = spatial.Delaunay(xs)
                    fit = interpolate.LinearNDInterpolator(tri, ys)
                    xis_test = []
                    for i in range(dimension):
                        l, h = ranges[i][0], ranges[i][1]
                        c, w = (h + l) / 2, (h - l) * TEST_COMPACTIFICATION
                        l, h = c - w / 2, c + w / 2
                        xis_test.append(np.random.uniform(low=l, high=h, size=(N_TEST_POINTS,)))
                    xs_test = np.array(xis_test).T
                    ys_test = target_fn(*[xis_test[i] for i in range(dimension)])
                    
                    preds = fit(xs_test)
                    valid_idx = ~np.isnan(preds)
                    rmse_loss = np.sqrt(np.mean(np.power(preds[valid_idx] - ys_test[valid_idx], 2))).item()
                    ex.info[eqn]['random'][D].append(rmse_loss) 

