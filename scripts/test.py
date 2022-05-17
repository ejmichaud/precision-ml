#!/usr/bin/env python
# coding: utf-8

import time

from sacred import Experiment
# from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("sacred-test")
# ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def cfg():
    pass


@ex.automain
def run(seed):
    time.sleep(11)
    print(ex.observers)
    ex.info["test"] = True
    # print(ex.observers[0])

