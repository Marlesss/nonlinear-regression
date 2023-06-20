import numpy as np
from typing import Callable

FUNC_USAGE = 0
GRAD_USAGE = 0


def curve(args: np.ndarray):
    def apply(x: float) -> float:
        n = len(args)
        return sum(args[i] * (x ** i) for i in range(n))

    return apply


def func(curve: Callable[[float], float]):
    def apply(x: float, y: float) -> float:
        global FUNC_USAGE
        FUNC_USAGE += 1
        return (y - curve(x)) ** 2

    return apply


def grad_func(args: np.ndarray, learning_rate: float):
    def apply(x: float, y: float) -> np.ndarray:
        global GRAD_USAGE
        GRAD_USAGE += 1
        n = len(args)
        this_curve = curve(args)
        return np.array([
            (-2 * y * (x ** i) +
             2 * this_curve(x) * (x ** i))
            for i in range(n)
        ])

    return apply


def mistake(func: Callable[[float], float], dots):
    return sum(func(x, y) ** (1 / 2) for x, y in dots)
