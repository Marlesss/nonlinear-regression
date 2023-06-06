from typing import Callable
import numpy as np
from math import sqrt

ITER_LIMIT = 50
EPS = 10 ** -3
CONVERGENCE_EPS = 1 / 2
PHI = (1 + 5 ** 0.5) / 2


def first_wolfe_condition(x: np.ndarray, gradient_x: np.ndarray, direction: np.ndarray, f_x: float, alpha: float,
                          c1: float, f: Callable[[np.ndarray], float]) -> bool:
    return f(x + alpha * direction) <= f_x + c1 * alpha * np.dot(gradient_x, direction)


def second_wolfe_condition(x: np.ndarray, gradient_x: np.ndarray, direction: np.ndarray, alpha: float,
                           c2: float, grad: Callable[[np.ndarray], np.ndarray]) -> bool:
    return np.dot(grad(x + alpha * direction), direction) >= c2 * np.dot(gradient_x, direction)


def golden_section_method_with_wolfe_conditions(x: np.ndarray, grad: Callable[[np.ndarray], np.ndarray],
                                                f: Callable[[np.ndarray], float], grad_value: np.array,
                                                c1: float, c2: float) -> float:
    left = 0
    right = 1
    f_x = f(x)
    direction = -grad_value
    med_left = left + (right - left) / (PHI + 1)
    med_right = right - (right - left) / (PHI + 1)
    f_left = f(x + med_left * direction)
    f_right = f(x + med_right * direction)

    while abs(left - right) > EPS:
        if f_left < f_right:
            right = med_right
            med_right = med_left
            f_right = f_left
            med_left = left + (right - left) / (PHI + 1)
            f_left = f(x + med_left * direction)
        else:
            left = med_left
            med_left = med_right
            f_left = f_right
            med_right = right - (right - left) / (PHI + 1)
            f_right = f(x + med_right * direction)
        checkpoint = (left + right) / 2
        if (first_wolfe_condition(x, grad_value, direction, f_x, checkpoint, c1, f)
                and second_wolfe_condition(x, grad_value, direction, checkpoint, c2, grad)):
            return checkpoint

    return left


def golden_section_method(x: np.ndarray, f: Callable[[np.ndarray], float], grad_value: np.ndarray) -> float:
    left = 0
    right = 1
    direction = -grad_value
    med_left = left + (right - left) / (PHI + 1)
    med_right = right - (right - left) / (PHI + 1)
    f_left = f(x + med_left * direction)
    f_right = f(x + med_right * direction)

    while abs(left - right) > EPS:
        if f_left < f_right:
            right = med_right
            med_right = med_left
            f_right = f_left
            med_left = left + (right - left) / (PHI + 1)
            f_left = f(x + med_left * direction)
        else:
            left = med_left
            med_left = med_right
            f_left = f_right
            med_right = right - (right - left) / (PHI + 1)
            f_right = f(x + med_right * direction)
    return left


def gradient_descent_linear_with_wolfe_condition(x0: np.ndarray, grad: Callable[[np.ndarray], np.ndarray],
                                                 f: Callable[[np.ndarray], float], c1: float, c2: float) \
        -> (bool, np.ndarray):
    dots = np.array([x0])
    prev_x = x0
    for _ in range(ITER_LIMIT):
        grad_value = grad(prev_x)
        if sqrt((grad_value ** 2).sum()) < EPS:
            return True, dots
        new_x = prev_x - golden_section_method_with_wolfe_conditions(prev_x, grad, f, grad_value, c1, c2) * grad_value
        dots = np.append(dots, [new_x], 0)
        prev_x = new_x
    return False, dots


def gradient_descent_linear(x0: np.ndarray, grad: Callable[[np.ndarray], np.ndarray],
                            f: Callable[[np.ndarray], float]) -> (bool, np.ndarray):
    dots = np.array([x0])
    prev_x = x0
    for _ in range(ITER_LIMIT):
        grad_value = grad(*prev_x)
        if sqrt((grad_value ** 2).sum()) < CONVERGENCE_EPS:
            return True, dots
        new_x = prev_x - golden_section_method(prev_x, f, grad_value) * grad_value
        dots = np.append(dots, [new_x], 0)
        prev_x = new_x
    return False, dots


def gradient_descent_constant(x0: np.ndarray, alph: float,
                              grad: Callable[[np.ndarray], np.ndarray]) -> (bool, [np.ndarray]):
    dots = np.array([x0])
    prev_x = x0
    for _ in range(ITER_LIMIT):
        grad_value = grad(*prev_x)
        if sqrt((grad_value ** 2).sum()) < CONVERGENCE_EPS:
            return True, dots
        new_x = prev_x - alph * grad_value
        dots = np.append(dots, [new_x], 0)
        prev_x = new_x
    return False, dots
