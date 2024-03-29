from typing import Callable

import numpy as np
import math
import random

EPS = 1e-5


def vectorize(func):
    def wrapped(args: np.ndarray):
        return func(*args)

    return wrapped


def generate_dots_on_line(n: int, line: Callable[[float], float], noize: float = 0, x_from=0, x_to=100) -> np.ndarray:
    return np.array([(x, line(x) + (random.random() - 1 / 2) * noize) for x in np.linspace(x_from, x_to, n)])


def r(g: Callable[[float], float], g_difs: [Callable[[float], float]]):
    def apply(x: float, y: float):
        return y - g(x)

    def dif(g_dif: Callable[[float], float]):
        def apply_dif(x: float, y: float):
            return -g_dif(x)

        return apply_dif

    return apply, [dif(g_dif) for g_dif in g_difs]


def jacobi(dots: [(float, float)], g: Callable[[float], float], g_difs: [Callable[[float], float]]):
    r_apply, r_difs = r(g, g_difs)
    return [[r_dif(x, y) for r_dif in r_difs] for (x, y) in dots]


def hessian(dots: [(float, float)], g: Callable[[float], float], g_difs: [Callable[[float], float]]):
    J = np.array(jacobi(dots, g, g_difs))
    return J.T.dot(J)


def f(dots: [(float, float)], g_factory):
    def apply(args: [float]):
        g, g_difs = g_factory(*args)
        r_apply, _ = r(g, g_difs)
        return sum(r_apply(x, y) ** 2 for (x, y) in dots)

    return apply


grad_f_apply_count = 0


def grad_f(jacobian: np.ndarray, r_vector: np.ndarray):
    global grad_f_apply_count
    grad_f_apply_count += 1
    return np.dot(jacobian.T, r_vector)


def grad_f_get(dots, g_factory):
    def grad(*args):
        g, g_difs = g_factory(*args)
        r_apply, _ = r(g, g_difs)
        r_vector = np.array([r_apply(x, y) for (x, y) in dots])
        jacobian = np.array(jacobi(dots, g, g_difs))
        return grad_f(jacobian, r_vector)

    return grad


def grad_square_f(jacobian: np.ndarray):
    return np.dot(jacobian.T, jacobian)


g_exponent_apply_count = 0


def g_exponent(a: float, b: float):
    # a * e^(b * x)

    def apply(x: float):
        global g_exponent_apply_count
        g_exponent_apply_count += 1
        return a * math.e ** (b * x)

    def dif_a(x: float):
        return math.e ** (b * x)

    def dif_b(x: float):
        return a * x * math.e ** (b * x)

    return apply, [dif_a, dif_b]


def g_sin(a: float):
    def apply(x: float):
        return math.sin(a * x)

    def dif_a(x: float):
        return math.cos(a * x) * x

    return apply, [dif_a]


g_2sin_apply_count = 0


def g_2sin(a: float, b: float):
    def apply(x: float):
        global g_2sin_apply_count
        g_2sin_apply_count += 1
        return a * math.sin(b * x)

    def dif_a(x: float):
        return math.sin(b * x)

    def dif_b(x: float):
        return a * x * math.cos(b * x)

    return apply, [dif_a, dif_b]


def g_2parabola(a: float, b: float):
    def apply(x: float):
        return a * x ** 2 + b * x

    def dif_a(x: float):
        return x ** 2

    def dif_b(x: float):
        return x

    return apply, [dif_a, dif_b]


def mistake(dots: [(float, float)], g: Callable[[float], float]):
    return sum(abs(y - g(x)) for (x, y) in dots)
