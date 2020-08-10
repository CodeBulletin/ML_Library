import cupy as cp


class TanH:
    @staticmethod
    def fn(x):
        return cp.tanh(x)

    def d(self, x):
        return 1 - self.fn(x) ** 2


class Sigmoid:
    @staticmethod
    def fn(x):
        return 1/(1 + cp.exp(-x))

    def d(self, x):
        y = self.fn(x)
        return y*(1-y)


class ReLu:
    @staticmethod
    def fn(x):
        return cp.maximum(0, x)

    @staticmethod
    def d(x):
        x[x >= 0] = 1
        x[x <= 0] = 0
        return x


class LReLU:
    @staticmethod
    def fn(x):
        return cp.maximum(0.01 * x, x)

    @staticmethod
    def d(x):
        x[x >= 0] = 1
        x[x <= 0] = 0.01
        return x


class PReLU:
    def __init__(self, param):
        self.param = param

    def fn(self, x):
        return cp.maximum(self.param * x, x)

    def d(self, x):
        x[x >= 0] = 1
        x[x <= 0] = self.param
        return x


class ELU:
    def __init__(self, param):
        self.param = param

    def fn(self, x):
        return cp.maximum(self.param * (cp.exp(-x) - 1), x)

    def d(self, x):
        return cp.where(x > 0, cp.ones_like(x), self.param * cp.exp(x))


class SoftPlus:
    @staticmethod
    def fn(x):
        return cp.log(cp.exp(x) + 1)

    @staticmethod
    def d(x):
        exp_x = cp.exp(x)
        return exp_x/(exp_x+1)


class SoftMax:
    @staticmethod
    def fn(x):
        t = cp.exp(x - cp.max(x))
        return t / cp.sum(t, axis=0, keepdims=True)

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
