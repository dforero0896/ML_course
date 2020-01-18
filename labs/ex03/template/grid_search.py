# -*- coding: utf-8 -*-
""" Grid Search"""

import numpy as np
import costs

def compute_loss(y, tx, w, kind = 'mse'):
    """Calculate the loss.
    You can calculate the loss using mse or mae.
    """
    error = y - tx.dot(w)
    if kind == 'mse':
        return error.dot(error)/(2*len(y))
    elif kind == 'mae':
        return sum(abs(error))/(2*len(y))
    else:
        raise NotImplementedError

def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


def grid_search(y, tx, w0, w1, kind = 'mse'):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    for i,w0_ in enumerate(w0):
        for j,w1_ in enumerate(w1):
            losses[i,j] = compute_loss(y, tx, [w0_, w1_], kind=kind)
    return losses
