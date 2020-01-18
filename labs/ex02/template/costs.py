# -*- coding: utf-8 -*-
"""Function used to compute the loss."""

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
