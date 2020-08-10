import numpy as np


class TanH:
    @staticmethod
    def fn(x):
        return np.tanh(x)

    def d(self, x):
        return 1 - self.fn(x) ** 2


class Sigmoid:
    @staticmethod
    def fn(x):
        return 1/(1+np.exp(-x))

    def d(self, x):
        y = self.fn(x)
        return y*(1-y)


class ReLu:
    @staticmethod
    def fn(x):
        return np.maximum(0, x)

    @staticmethod
    def d(x):
        x[x >= 0] = 1
        x[x <= 0] = 0
        return x


class LReLU:
    @staticmethod
    def fn(x):
        return np.maximum(0.01*x, x)

    @staticmethod
    def d(x):
        x[x >= 0] = 1
        x[x <= 0] = 0.01
        return x


class PReLU:
    def __init__(self, param):
        self.param = param

    def fn(self, x):
        return np.maximum(self.param*x, x)

    def d(self, x):
        x[x >= 0] = 1
        x[x <= 0] = self.param
        return x


class ELU:
    def __init__(self, param):
        self.param = param

    def fn(self, x):
        return np.maximum(self.param*(np.exp(-x)-1), x)

    def d(self, x):
        return np.where(x > 0, np.ones_like(x), self.param*np.exp(x))


class SoftPlus:
    @staticmethod
    def fn(x):
        return np.log(np.exp(x)+1)

    @staticmethod
    def d(x):
        exp_x = np.exp(x)
        return exp_x/(exp_x+1)


class SoftMax:
    @staticmethod
    def fn(x):
        t = np.exp(x - np.max(x))
        return t/np.sum(t, axis=0, keepdims=True)

    @staticmethod
    def d(x, y):
        return x-y


class LinearActivation:
    @staticmethod
    def fn(x):
        return x

    @staticmethod
    def d(_):
        return 1
