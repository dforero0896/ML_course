# -*- coding: utf-8 -*-
"""Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w, kind='mse'):
    """Compute a stochastic gradient from just few examples n and their
    corresponding y_n labels."""
    error = y - tx.dot(w)
    if kind == 'mse':
        return -tx.T.dot(error) / len(y)
    elif kind == 'mae':
        return -np.sign(error).dot(tx) / len(y)
    else:
        raise NotImplementedError

def stochastic_gradient_descent(y,
                                tx,
                                initial_w,
                                batch_size,
                                max_iters,
                                gamma,
                                kind='mse'):
    """Stochastic gradient descent algorithm."""
    # implement stochastic gradient descent
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w.astype(float)
    for n_iter in range(max_iters):
        for new_y, new_tx in batch_iter(y,
                                        tx,
                                        batch_size=batch_size,
                                        num_batches=1):
            # compute gradient and loss
            gradient = compute_stoch_gradient(new_y, new_tx, w, kind=kind)
            loss = compute_loss(new_y, new_tx, w, kind=kind)
            # update w by gradient
            #gamma = 1./(n_iter + 1)
            w = w - gamma * gradient
            # store w and loss
            ws.append(w)
            losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def stochastic_gradient_descent_single(
        y, tx, initial_w, batch_size, max_iters, gamma, kind = 'mse'):
    """Stochastic gradient descent algorithm."""
    # implement stochastic gradient descent
    indices = np.arange(len(y))
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w.astype(float)
    for n_iter in range(max_iters):
        np.random.shuffle(indices)
        new_y = y[indices[:batch_size]]
        new_tx = tx[indices[:batch_size],:]
        # compute gradient and loss
        gradient = compute_stoch_gradient(new_y, new_tx, w, kind = kind)
        loss = compute_loss(new_y, new_tx, w, kind = kind)
        # update w by gradient
        #gamma = 1./(n_iter + 1)
        w = w - gamma * gradient
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
