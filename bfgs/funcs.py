import numpy as np


def norm():
    def func(x1: float, x2: float):
        return (x1 + 10) ** 2 + 5 * (x2 + 10) ** 2

    def grad(x1: float, x2: float):
        return np.array([2 * (x1 + 10), 2 * 5 * (x2 + 10)])

    return func, grad


def bad():
    def func(x1: float, x2: float):
        return 10 * (x1 - 5) ** 2 + (x2 + 10) ** 2

    def grad(x1: float, x2: float):
        return np.array([2 * 10 * (x1 - 5), 2 * (x2 + 10)])

    return func, grad
