# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    gram_matrix = tx.transpose().dot(tx)
    # Solve for w using a solver. Just multiplying by the inverse yields incorrect results in some cases
    w = np.linalg.solve(gram_matrix, tx.transpose().dot(y))
    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e))
    return w, loss
