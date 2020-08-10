import numpy as np


def LinearScaling(x):
    return (x + np.min(x)) / (np.max(x) - np.min(x))


def logScaling(x):
    return np.log(x)


def Z_score(x):
    n = len(x[0])
    u = (1/n)*np.sum(x, axis=1, keepdims=True)
    sigma = np.sqrt((1/(n-1))*np.sum((x-u)**2, axis=1, keepdims=True))
    return (x-u)/sigma
