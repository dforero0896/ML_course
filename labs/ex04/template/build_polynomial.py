# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.array(list(map(lambda j: np.array(x)**j, range(degree+1))), dtype=float)
    return phi.T
