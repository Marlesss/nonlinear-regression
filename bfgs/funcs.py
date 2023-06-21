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


rosenbrock_func_apply_cnt, rosenbrock_grad_apply_cnt = 0, 0


def rosenbrock():
    def func(x1: float, x2: float):
        global rosenbrock_func_apply_cnt
        rosenbrock_func_apply_cnt += 1
        return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2

    def grad(x1: float, x2: float):
        global rosenbrock_grad_apply_cnt
        rosenbrock_grad_apply_cnt += 1
        return np.array([
            -2 + 2 * x1 - 400 * x1 * x2 + 400 * x1 ** 3,
            200 * (x2 - x1 ** 2)
        ])

    return func, grad
