# -*- coding: utf-8 -*-
"""Gradient Descent"""

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
    ws = [initial_w]
    losses = []
    w = initial_w.astype(float)
    for n_iter in range(max_iters):
        # compute gradient and loss
        gradient = compute_gradient(y, tx, w, kind=kind)
        loss = compute_loss(y, tx, w, kind=kind)
        # update w by gradient
        #gamma = 1./(n_iter + 1)
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws
