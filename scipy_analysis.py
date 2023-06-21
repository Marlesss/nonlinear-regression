import time

import scipy
import math_func
import numpy as np
from typing import Callable

import gauss_newton.nonlinear_regression


def run_with_statistics(stats: [Callable[[], float]], runnable, ans_completion):
    before_stats = [stat_func() for stat_func in stats]
    ans = runnable()
    after_stats = [stat_func() for stat_func in stats]
    ans_completion(ans)
    work_stats = [after_stats[i] - before_stats[i] for i in range(len(stats))]
    print(f"Work stats is {work_stats}")


def show_ans(dots, g_fact, ans: np.ndarray):
    print(f"Solution is {ans}")
    print(f"Mistake of solution is {math_func.mistake(dots, g_fact(*ans)[0])}")


def show_scipy_ans(dots, g_fact, ans: scipy.optimize.OptimizeResult):
    print(ans)
    show_ans(dots, g_fact, ans["x"])


def main():
    stats = [
        lambda: math_func.g_exponent_apply_count,
        lambda: math_func.grad_f_apply_count,
        lambda: time.time()
    ]
    g_fact = math_func.g_exponent
    correct = 2., 2.
    start = np.array([0.99, 1.6])

    g_apply, difs = g_fact(*correct)
    dots = math_func.generate_dots_on_line(100, g_apply, 0, x_from=0, x_to=3)

    print("MY GAUSS_NEWTON")
    run_with_statistics(stats, lambda: gauss_newton.nonlinear_regression.gauss_newton(dots, start, g_fact),
                        lambda ans: show_ans(dots, g_fact, ans))
    print("________________")
    print("SCIPY NEWTON_CG")
    run_with_statistics(stats,
                        lambda: scipy.optimize.minimize(math_func.f(dots, g_fact), start, method="Newton-CG",
                                                        jac=math_func.vectorize(math_func.grad_f_get(dots, g_fact))),
                        lambda ans: show_scipy_ans(dots, g_fact, ans))

    # my_solution = gauss_newton.nonlinear_regression.gauss_newton(dots, start, g_fact)
    # print(f"My solution: {my_solution}")
    # print(f"Mistake of my solution: {mistake(dots, g_fact(*my_solution)[0])}")
    # scipy_ans = scipy.optimize.minimize(f(dots, g_fact), start, method="Newton-CG",
    #                                     jac=vectorize(grad_f_get(dots, g_fact)))
    # scipy_solution = scipy_ans["x"]
    # print(f"Scipy solution: {scipy_solution}")
    # print(f"Mistake of scipy solution: {mistake(dots, g_fact(*scipy_solution)[0])}")


if __name__ == "__main__":
    main()
