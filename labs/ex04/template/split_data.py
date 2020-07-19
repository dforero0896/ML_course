# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio.
    Returns: x_train, y_train, x_test, y_test
    """
    # Maximum number of elements to include in the train set
    max_ind = int(ratio * len(x))
    # set seed
    np.random.seed(seed)
    # Generate an ordered list of indices
    ids = np.arange(len(x))
    # Suffle indices
    np.random.shuffle(ids)
    #Get shuffled arrays
    x_shuf = x[ids]
    y_shuf = y[ids]
    # Return both subsets
    return x_shuf[:max_ind], y_shuf[:max_ind], x_shuf[max_ind:], y_shuf[max_ind:]


