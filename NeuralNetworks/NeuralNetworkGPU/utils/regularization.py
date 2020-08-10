import cupy as cp


class L2:
    @staticmethod
    def fn(weights, biases):
        x1 = [cp.sum(i ** 2) for i in weights]
        x2 = [cp.sum(i ** 2) for i in biases]
        y1 = sum(x1)
        y2 = sum(x2)
        return y1 + y2

    @staticmethod
    def fn2(x, a, b):
        return a/b * x
