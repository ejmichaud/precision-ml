# precision-ml

This repository contains code for the paper `Precision Machine Learning`, by [Eric J. Michaud](https://ericjmichaud.com), [Ziming Liu](https://kindxiaoming.github.io/), and [Max Tegmark](https://space.mit.edu/home/tegmark/home.html).

## Organization

Within `notebooks/create-paper-figures` you can find the code which generated all the figures in the paper. There are some other jupyter notebooks within `notebooks` to replicate other experiments mentioned in the paper which do not have a dedicated figure.

Many experiments involved running a grid search over things like network width, target equation, seed, activation function, etc. Configs defining these grid searches can be found in `seml-experiments`. We used the [seml](https://github.com/TUM-DAML/seml) package for executing these experiments, which call scripts (which are built with [sacred](https://github.com/IDSIA/sacred)) in the `scripts` directory.

`equations.csv` is a modified version of `FeynmanEquations.csv` from Tegmark's [Feynman Symbolic Regression Database](https://space.mit.edu/home/tegmark/aifeynman.html), with some fixes and added equations.



