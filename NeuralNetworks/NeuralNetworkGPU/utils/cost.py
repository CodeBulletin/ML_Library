import cupy as cp


class MSE:
    @staticmethod
    def cost(x, y):
        return (x - y)**2

    @staticmethod
    def d_cost(x, y):
        return 2 * (x - y)


class MAE:
    @staticmethod
    def cost(x, y):
        return cp.abs(x - y)

    @staticmethod
    def d_cost(x, y):
        h = (x-y)
        h[h <= 0] = -1
        h[h >= 0] = 1
        return h


class CrossEntropy:
    @staticmethod
    def cost(x, y):
        return -(y * cp.log(x))

    @staticmethod
    def d_cost(x, y):
        return -(y/(x+1e-8) + (1-y)/((1-x)+1e-8))
