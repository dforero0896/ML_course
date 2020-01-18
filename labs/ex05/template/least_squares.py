# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_loss(y, tx, w, kind='mse'):
    error = y - tx.dot(w)
    if kind == 'mse':
        return error.dot(error)/(2*len(y))
    elif kind == 'mae':
        return sum(np.abs(error))/len(y)
    else:
        raise NotImplementedError
def least_squares(y, tx):
    """calculate the least squares solution."""
    gram_matrix = tx.transpose().dot(tx)
    # Solve for w using a solver. Just multiplying by the inverse yields incorrect results in some cases
    w = np.linalg.solve(gram_matrix, tx.transpose().dot(y))
    e = y - tx.dot(w)
    loss = compute_loss(y, tx, w)
    return w, loss