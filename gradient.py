import time

import math
import numpy as np
from typing import Callable
import torch
import numdifftools as ndt
import pprint


def default_grad(func: Callable[[np.ndarray], float], x: np.ndarray, h=1e-6):
    grad = np.zeros(len(x))
    x_mod = np.array(x, dtype=float)
    for i in range(len(x)):
        x_mod[i] += h
        grad[i] = (func(x_mod) - func(x)) / h
        x_mod[i] -= h
    return grad


def torch_grad(func: Callable[[np.ndarray], float], x: np.ndarray):
    x_tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y = func(x_tensor)
    y.backward()
    return np.array(x_tensor.grad)


def numpy_grad(func: Callable[[np.ndarray], float], x: np.ndarray, h=1e-6):
    results = {"np": np}
    exec(f'grid = np.mgrid[{", ".join(f"{x_i - h}:{x_i + h}:{h}" for x_i in x)}]', results)
    grid = results["grid"]
    res = np.array(np.gradient(func(grid), h))
    if len(x) == 1:
        return np.array([res[1]])
    else:
        return res[:, 1, 1]


def ndt_grad(func: Callable[[np.ndarray], float], x: np.ndarray):
    res = ndt.core.Gradient(func)(x)
    if not res.shape:
        res = np.array([res])
    return res


if __name__ == "__main__":
    def to_fixed(num, digits=6):
        return f"{num:.{digits}f}"


    for func_name, f, x_coll in [
        ("x^2", lambda x: x ** 2, [np.array([i]) for i in range(-3, 4)]),
        ("x^2 + xy + y^2", lambda x: x[0] ** 2 + x[0] * x[1] + x[1] ** 2,
         [np.array([i, j]) for i in range(-2, 3, 2) for j in range(-2, 3, 2)]),
        ("x * e ^ y", lambda x: x[0] * math.e ** x[1], [np.array([x, y]) for x in range(0, 3) for y in range(0, 3)]),
        ("(1 - x) ^ 2 + 100(y - x^2)^2", lambda x: (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2,
         [np.array([x, y]) for x in range(-1, 2) for y in range(-1, 2)])
    ]:
        print(f"func: {func_name}")
        print(f"dots: {list(map(list, x_coll))}")
        for grad in [default_grad, torch_grad, numpy_grad, ndt_grad]:
            t_start = time.time()
            pprint.pprint([list(map(float, map(to_fixed, grad(f, x)))) for x in x_coll])
            t_end = time.time()
            t = t_end - t_start
            print(f"time = {t}")
        print("_" * 40)
