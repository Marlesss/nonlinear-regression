import numpy as np
from random import shuffle
from typing import Callable
import stochastic_gradient_descent.func_utils as func_utils
import math

CONVERGENCE_EPS = 1e-2
EPS = 10 ** -8


def default_new_args(prev_args: np.ndarray, learning_rate: float, grad: np.ndarray) -> np.ndarray:
    return prev_args - learning_rate * grad


def constant_learning_rate():
    def func(init_learning_rate: float, epoch: int):
        return init_learning_rate

    return func


def linear_learning_rate_scheduler(schedule_coef: float, epoch_step: int) -> Callable[[float, int], float]:
    def func(init_learning_rate: float, epoch: int) -> float:
        return init_learning_rate * math.pow(schedule_coef, (1 + epoch) // epoch_step)

    return func


def exponent_learning_rate_scheduler(schedule_coef: float) -> Callable[[float, int], float]:
    def func(init_learning_rate: float, epoch: int) -> float:
        return init_learning_rate * math.exp(-schedule_coef * epoch)

    return func


def stochastic_factory(batch_size: int, init_learning_rate: float,
                       learning_rate_scheduler: Callable[[float, int], float],
                       grad_func_factory: Callable[[np.ndarray, float], Callable[[float, float], np.ndarray]],
                       new_args_func: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
                       epoch_limit: int = 50,
                       log: int = 0):
    def apply(dots: np.ndarray, start_value: np.ndarray) -> (bool, np.ndarray):
        k = len(dots)
        n = len(start_value)
        dots_i = list(range(k))

        way = [start_value]
        prev_args = start_value

        converged = False
        for epoch in range(epoch_limit):
            if converged:
                break
            learning_rate = learning_rate_scheduler(init_learning_rate, epoch)
            shuffle(dots_i)
            for i in range((k + batch_size - 1) // batch_size):
                minibatch_i = dots_i[i * batch_size: (i + 1) * batch_size]
                minibatch_dots = [dots[i] for i in minibatch_i]

                grad_func = grad_func_factory(prev_args, learning_rate)
                grad = np.array([0.0] * n)
                for dot in minibatch_dots:
                    grad += grad_func(dot[0], dot[1])

                if (abs(grad) < CONVERGENCE_EPS).all():
                    converged = True
                    break

                new_args = new_args_func(prev_args, learning_rate, grad)
                way.append(new_args)
                prev_args = new_args
            if log and epoch % log == 0:
                print(f"epoch {epoch}. args: {prev_args}")
        return converged, np.array(way)

    return apply


def momentum_mod(gamma: float) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
    prev_v = 0

    def apply(prev_args: np.ndarray, learning_rate: float, grad: np.ndarray) -> np.ndarray:
        nonlocal prev_v
        v = gamma * prev_v + (1 - gamma) * grad
        new_args = prev_args - learning_rate * v
        prev_v = v
        return new_args

    return apply


def nesterov_mod(gamma: float,
                 grad_func: Callable[[np.ndarray, float], Callable[[float, float], np.ndarray]]) \
        -> (Callable[[np.ndarray, float], Callable[[float, float], np.ndarray]],
            Callable[[np.ndarray, float, np.ndarray], np.ndarray]):
    prev_v = 0

    def grad_func_factory(prev_args: np.ndarray, learning_rate: float) -> Callable[[float, float], np.ndarray]:
        nonlocal prev_v
        return grad_func(prev_args - learning_rate * gamma * prev_v, learning_rate)

    def new_args_factory(prev_args: np.ndarray, learning_rate: float, grad: np.ndarray) -> np.ndarray:
        nonlocal prev_v
        v = gamma * prev_v + (1 - gamma) * grad
        new_args = prev_args - learning_rate * v
        prev_v = v
        return new_args

    return grad_func_factory, new_args_factory


def adagrad_mod() -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
    G = EPS

    def apply(prev_args: np.ndarray, learning_rate: float, grad: np.ndarray) -> np.ndarray:
        nonlocal G
        G += grad ** 2
        return prev_args - learning_rate * ((G + EPS) ** (-1 / 2)) * grad

    return apply


def rmsprop_mod(beta: float) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
    prev_s = EPS

    def apply(prev_args: np.ndarray, learning_rate: float, grad: np.ndarray) -> np.ndarray:
        nonlocal prev_s
        s = beta * prev_s + (1 - beta) * (grad ** 2)
        prev_s = s
        return prev_args - learning_rate * grad * ((s + EPS) ** (-1 / 2))

    return apply


def adam_mod(beta1: float, beta2: float) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
    beta1_powered = beta1
    beta2_powered = beta2
    prev_v = 0
    prev_s = 0

    def apply(prev_args: np.ndarray, learning_rate: float, grad: np.ndarray) -> np.ndarray:
        nonlocal beta1_powered
        nonlocal beta2_powered
        nonlocal prev_v
        nonlocal prev_s
        v = beta1 * prev_v + (1 - beta1) * grad
        s = beta2 * prev_s + (1 - beta2) * (grad ** 2)
        new_args = (prev_args - learning_rate * (v / (1 - beta1_powered)) * (s / (1 - beta2_powered) + EPS) ** (-1 / 2))
        prev_v = v
        prev_s = s
        beta1_powered *= beta1
        beta2_powered *= beta2
        return new_args

    return apply


def l1_regu(new_args_func: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
            coef: float) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
    def apply(prev_args: np.ndarray, learning_rate: float, grad: np.ndarray) -> np.ndarray:
        grad += coef * np.array([1 if arg > 0 else -1 for arg in prev_args])
        return new_args_func(prev_args, learning_rate, grad)

    return apply


def l2_regu(new_args_func: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
            coef: float) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
    def apply(prev_args: np.ndarray, learning_rate: float, grad: np.ndarray) -> np.ndarray:
        grad += coef * 2 * prev_args
        return new_args_func(prev_args, learning_rate, grad)

    return apply


def elastic(new_args_func: Callable[[np.ndarray, float, np.ndarray], np.ndarray],
            coef: float, l1_coef: float, l2_coef: float) -> Callable[[np.ndarray, float, np.ndarray], np.ndarray]:
    def apply(prev_args: np.ndarray, learning_rate: float, grad: np.ndarray) -> np.ndarray:
        l1_regr = l1_coef * np.array([1 if arg > 0 else -1 for arg in prev_args])
        l2_regr = l2_coef * 2 * prev_args
        grad += coef * l1_regr + (1 - coef) / 2 * l2_regr
        return new_args_func(prev_args, learning_rate, grad)

    return apply


def polynomial_stochastic_gradient_descent(dots: np.ndarray, batch_size: int, start_value: np.ndarray,
                                           learning_rate: float,
                                           l1: float = 0, l2: float = 0, elastic: float = 0, epoch_limit=2000):
    way = [start_value]
    n = len(start_value)
    dots_i = list(range(len(dots)))
    prev_args = start_value

    converges = False
    for epoch in range(epoch_limit):
        if converges:
            break
        shuffle(dots_i)
        for i in range((len(dots) + batch_size - 1) // batch_size):
            minibatch_i = dots_i[i * batch_size: (i + 1) * batch_size]
            minibatch_dots = [dots[i] for i in minibatch_i]
            grad = np.array([0.0] * n)
            grad_func = func_utils.grad_func(prev_args, learning_rate)
            for dot in minibatch_dots:
                grad += grad_func(dot[0], dot[1])

            l1_regr = l1 * np.array([1 if arg > 0 else -1 for arg in prev_args])
            l2_regr = l2 * 2 * prev_args
            if abs(elastic) > EPS:
                grad += elastic * l1_regr + (1 - elastic) / 2 * l2_regr
            else:
                grad += l1_regr + l2_regr

            new_args = prev_args - learning_rate * grad
            if all(abs(new_args - prev_args) < CONVERGENCE_EPS):
                converges = True
                break
            way.append(new_args)
            prev_args = new_args

    return np.array(way)
