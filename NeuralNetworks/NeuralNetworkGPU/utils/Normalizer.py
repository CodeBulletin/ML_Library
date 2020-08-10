import cupy as cp


def LinearScaling(x):
    return (x + cp.min(x)) / (cp.max(x) - cp.min(x))


def logScaling(x):
    return cp.log(x)


def Z_score(x):
    n = len(x[0])
    u = (1/n) * cp.sum(x, axis=1, keepdims=True)
    sigma = cp.sqrt((1 / (n - 1)) * cp.sum((x - u) ** 2, axis=1, keepdims=True))
    return (x-u)/sigma
