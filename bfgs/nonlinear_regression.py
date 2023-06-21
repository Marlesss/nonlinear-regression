import numpy as np
from math_func import *

PHI = (1 + 5 ** 0.5) / 2


def first_wolfe_condition(x: np.ndarray, gradient_x: np.ndarray, direction: np.ndarray, f_x: float, alpha: float,
                          c1: float, f: Callable[[np.ndarray], float]) -> bool:
    return f(x + alpha * direction) <= f_x + c1 * alpha * np.dot(gradient_x, direction)


def second_wolfe_condition(x: np.ndarray, gradient_x: np.ndarray, direction: np.ndarray, alpha: float,
                           c2: float, grad: Callable[[np.ndarray], np.ndarray]) -> bool:
    return np.dot(grad(*(x + alpha * direction)), direction) >= c2 * np.dot(gradient_x, direction)


def golden_section_method_with_wolfe_conditions(x: np.ndarray, grad: Callable[[np.ndarray], np.ndarray],
                                                f: Callable[[np.ndarray], float], grad_value: np.array,
                                                c1=10 ** -6, c2=2 * 10 ** -6) -> float:
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


# def getCorrectAlpha(step, grad_step, p_step: np.ndarray, func, grad, c1=10 ** -6, c2=2 * 10 ** -6):
#     a = 0
#     b = 1
#     alpha = 1
#     firstCondition = c1 * p_step.dot(grad_step)
#     secondCondition = abs(c2 * p_step.dot(grad_step))
#     currentF = func(step)
#     for count in range(1, 100):
#         alpha = (a + b) / 2
#         supposedX = step + alpha * p_step
#         supposedF = func(supposedX)
#         if (supposedF <= currentF + alpha * firstCondition) and (
#                 abs(p_step.dot(np.asarray(grad(*supposedX)))) <= secondCondition):
#             break
#         f1, f2 = supposedF, func(supposedX - EPS * p_step)
#         if f1 < f2:
#             a = alpha
#         else:
#             b = alpha
#     return alpha


def BFGS(start, func, grad):
    iter_count = 0
    way = [start]
    grad_step = grad(*start)
    I = np.identity(len(start))
    H = I
    step = start
    while True:
        iter_count += 1
        p_step = -H.dot(grad_step)
        # alpha = getCorrectAlpha(step, grad_step, p_step, func, grad)
        alpha = golden_section_method_with_wolfe_conditions(step, grad, func, grad(*step))
        s_step = alpha * p_step
        next_step = step + s_step

        if np.linalg.norm(next_step - step) < EPS:
            break
        grad_next = grad(*next_step)
        y_step = grad_next - grad_step
        rho = 1 / y_step.dot(s_step)
        H = (I - rho * np.outer(s_step, y_step)).dot(H).dot(I - rho * np.outer(y_step, s_step)) \
            + rho * np.outer(s_step, s_step)
        way.append(next_step)
        step = next_step
        grad_step = grad_next
    return np.array(way), iter_count


def L_BFGS(start, func, grad, m=10):
    iter_count = 0
    way = [start]
    mem = []
    grad_step = grad(*start)
    step = start
    I = np.identity(len(start))
    while True:
        iter_count += 1
        if len(mem) == 0:
            H = I
        else:
            H = np.zeros_like(I)
            V_mult = I
            for (s, y) in mem[::-1]:
                rho = 1 / y.dot(s)
                s_mult = np.outer(s, s)
                H += rho * V_mult.T.dot(s_mult.dot(V_mult))
                V = I - rho * np.outer(y, s)
                V_mult = V.dot(V_mult)
            H += V_mult.T.dot(V_mult)
        p_step = -H.dot(grad_step)

        # alpha = getCorrectAlpha(step, grad_step, p_step, func, grad)
        alpha = golden_section_method_with_wolfe_conditions(step, grad, func, grad(*step))

        s_step = alpha * p_step
        next_step = step + s_step

        if np.linalg.norm(next_step - step) < EPS:
            break
        grad_next = grad(*next_step)
        y_step = grad_next - grad_step
        if len(mem) == m:
            del mem[0]
        mem.append((s_step, y_step))
        way.append(next_step)
        step = next_step
        grad_step = grad_next
    return np.array(way), iter_count
