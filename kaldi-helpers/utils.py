"""
These are helper function for calculations not directly related to kaldi itself.
"""

import numpy as np


def softmax(x: np.array) -> np.array:
    """Transform each element of an array by softmax"""
    x = x.T
    x = x - np.max(x, axis=0)
    x = np.exp(x)
    x /= np.sum(x, axis=0)
    return x.T
