import time

import math
import scipy

import math_func
import numpy as np
from typing import Callable
import powell_dog_leg.nonlinear_regression
import gauss_newton.nonlinear_regression
import bfgs.nonlinear_regression
import bfgs.funcs


def run_with_statistics(stats: [Callable[[], float]], runnable, ans_completion):
    before_stats = [stat_func() for stat_func in stats]
    ans = runnable()
    after_stats = [stat_func() for stat_func in stats]
    ans_completion(ans)
    work_stats = [after_stats[i] - before_stats[i] for i in range(len(stats))]
    print(f"Work stats is {work_stats}")


def show_ans(dots, g_fact, ans: np.ndarray, iter_count: int):
    print(f"Solution is {ans}")
    print(f"Count of iterations is {iter_count}")
    print(f"Mistake of solution is {math_func.mistake(dots, g_fact(*ans)[0])}")


def show_bfgs_ans(func, ans, iter_count):
    print(f"Solution is {ans}")
    print(f"Count of iterations is {iter_count}")
    print(f"Function value in the answer is {func(*ans)}")


def show_scipy_ans(dots, g_fact, ans: scipy.optimize.OptimizeResult):
    print(ans)
    show_ans(dots, g_fact, ans["x"], ans["nit"])


def main():
    # stats = [
    #     lambda: math_func.g_2sin_apply_count,
    #     lambda: math_func.grad_f_apply_count,
    #     lambda: time.time()
    # ]
    # # g_fact = math_func.g_exponent
    # g_fact = math_func.g_2sin
    # correct = 2., 1
    # start = np.array([0.4, 0.6])
    #
    # g_apply, difs = g_fact(*correct)
    # dots = math_func.generate_dots_on_line(100, g_apply, 0, x_from=-2 * math.pi, x_to=2 * math.pi)
    #
    # run_with_statistics(stats,
    #                     # lambda: gauss_newton.nonlinear_regression.gauss_newton(dots, start, g_fact),
    #                     lambda: powell_dog_leg.nonlinear_regression.powell_dog_leg(dots, start, g_fact),
    #                     lambda ans: show_ans(dots, g_fact, ans[0], ans[1]))
    # print("________________")
    # run_with_statistics(stats,
    #                     lambda: scipy.optimize.minimize(math_func.f(dots, g_fact), start, method="dogleg",
    #                                                     jac=math_func.vectorize(math_func.grad_f_get(dots, g_fact)),
    #                                                     hess=lambda args: math_func.hessian(dots, *g_fact(*args))
    #                                                     ),
    #                     lambda ans: show_scipy_ans(dots, g_fact, ans))

    stats = [
        lambda: bfgs.funcs.rosenbrock_func_apply_cnt,
        lambda: bfgs.funcs.rosenbrock_grad_apply_cnt,
        lambda: time.time()
    ]
    start = np.array([5, 5])
    f, g = bfgs.funcs.rosenbrock()
    run_with_statistics(stats,
                        lambda: bfgs.nonlinear_regression.BFGS(start, math_func.vectorize(f), g),
                        lambda ans: show_bfgs_ans(f, ans[0][-1], ans[1]))
    print("________________")
    run_with_statistics(stats,
                        lambda: scipy.optimize.minimize(math_func.vectorize(f), start, method="BFGS"),
                        lambda ans: print(ans) and show_bfgs_ans(f, ans["x"], ans["nit"]))


if __name__ == "__main__":
    main()
