# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    gram_matrix = tx.transpose().dot(tx)
    lambda_term = 2 * len(y) * lambda_ * np.identity(gram_matrix.shape[0])
    w = np.linalg.solve(gram_matrix + lambda_term, tx.transpose().dot(y))
    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e)) #+ lambda_ * w.dot(w)
    return w, loss
