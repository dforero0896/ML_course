# -*- coding: utf-8 -*-
"""Gradient Descent"""
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

def compute_gradient(y, tx, w, kind='mse'):
    """Compute the gradient."""
    error = y - tx.dot(w)
    if kind == 'mse':
        return -tx.T.dot(error)/len(y)
    elif kind == 'mae':
        return -np.sign(error).dot(tx)/len(y) #Sum rows
    else:
        raise NotImplementedError


def gradient_descent(y, tx, initial_w, max_iters, gamma, kind='mse'):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    w = initial_w.astype(float)
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w, kind=kind)
        loss = compute_loss(y, tx, w, kind=kind)
        # update w by gradient
        #gamma = 1./(n_iter + 1)
        w = w - gamma * gradient
    return loss, w
