# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    x = np.array(x)
    phi = np.array([x**i for i in range(degree+1)]).T
    return phi
