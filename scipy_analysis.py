import time

import math
import scipy
from scipy.optimize import LinearConstraint, NonlinearConstraint

import math_func
import numpy as np
from typing import Callable
import plot
import matplotlib.pyplot as plt
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


def show_bfgs_ans(func, ans, iter_count, jacob=None):
    print(f"Solution is {ans}")
    print(f"Count of iterations is {iter_count}")
    if jacob:
        print(f"Count of jacob is {jacob}")
    print(f"Function value in the answer is {func(*ans)}")


def show_scipy_ans(dots, g_fact, ans: scipy.optimize.OptimizeResult):
    # print(ans)
    show_ans(dots, g_fact, ans["x"], ans["nit"])


def show_way_graph(contour_func, correct, way, bounds=None, constraints=None):
    # print(f"WHOLE WAY IS {way}")
    plt.rcParams['figure.figsize'] = [10, 4]
    plt.subplot(1, 2, 1)
    x_min, x_max, y_min, y_max = (min(min(way[:, 0]), correct[0] - 1),
                                  max(max(way[:, 0]), correct[0] + 1),
                                  min(min(way[:, 1]), correct[1] - 1),
                                  max(max(way[:, 1]), correct[1] + 1))
    plt.plot(correct[0], correct[1], 'o', color=(0, 0, 0))
    if bounds:
        lb, ub = ([bounds.lb[0], bounds.lb[1]],
                  [bounds.ub[0], bounds.ub[1]])
        x_min, x_max, y_min, y_max = min(x_min, lb[0]), max(x_max, ub[0]), min(y_min, lb[1]), max(y_max, ub[1])
        bounds_way = np.array([
            [lb[0], lb[1]],
            [ub[0], lb[1]],
            [ub[0], ub[1]],
            [lb[0], ub[1]],
            [lb[0], lb[1]]
        ])
        plt.plot(bounds_way[:, 0], bounds_way[:, 1], "-", linewidth=0.8, color=(1, 1, 1))
    plot.show_2arg_func_slice(contour_func, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, show=False,
                              dots_show=False, contour=True, constraints=constraints)
    # plt.plot(way[:, 0], way[:, 1], '-', color=(1, 0, 0))
    plot.show_2arg_func(None, way, show=False)
    plot.show_2arg_func(None, way, quiver=True, width=0.01, alpha=1, show=False)

    plt.subplot(1, 2, 2)
    plot.show_2arg_func(contour_func, way, contour=True, quiver=True, label=None)


def loss_fn(dots, g_fact):
    return lambda args: math_func.mistake(dots, g_fact(*args)[0])


def bounds_analysis_Powell(dots, g_fact, correct, start, bounds, stats):
    way = [start]
    run_with_statistics(stats,
                        lambda: scipy.optimize.minimize(math_func.f(dots, g_fact), start, method="Powell",
                                                        jac=math_func.vectorize(math_func.grad_f_get(dots, g_fact)),
                                                        hess=lambda args: math_func.hessian(dots, *g_fact(*args)),
                                                        bounds=bounds,
                                                        callback=way.append
                                                        ),
                        lambda ans: (show_scipy_ans(dots, g_fact, ans),
                                     show_way_graph(loss_fn(dots, g_fact), correct, np.array(way), bounds=bounds)))


def bounds_analysis_BFGS(f, correct, start, bounds, stats):
    way = [start]
    run_with_statistics(stats,
                        lambda: scipy.optimize.minimize(math_func.vectorize(f), start, method="L-BFGS-B",
                                                        bounds=bounds,
                                                        callback=way.append),
                        lambda ans: (show_bfgs_ans(f, ans["x"], ans["nit"], jacob=ans['njev']),
                                     show_way_graph(math_func.vectorize(f), correct, np.array(way), bounds=bounds)))


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
    #                     lambda: gauss_newton.nonlinear_regression.gauss_newton(dots, start, g_fact),
    #                     # lambda: powell_dog_leg.nonlinear_regression.powell_dog_leg(dots, start, g_fact),
    #                     lambda ans: show_ans(dots, g_fact, ans[0], ans[1]))
    # print("________________")
    # run_with_statistics(stats,
    #                     lambda: scipy.optimize.minimize(math_func.f(dots, g_fact), start, method="dogleg",
    #                                                     jac=math_func.vectorize(math_func.grad_f_get(dots, g_fact)),
    #                                                     hess=lambda args: math_func.hessian(dots, *g_fact(*args))
    #                                                     ),
    #                     lambda ans: show_scipy_ans(dots, g_fact, ans))

    # stats = [
    #     lambda: bfgs.funcs.rosenbrock_func_apply_cnt,
    #     lambda: bfgs.funcs.rosenbrock_grad_apply_cnt,
    #     lambda: time.time()
    # ]
    # start = np.array([5, 5])
    # f, g = bfgs.funcs.rosenbrock()
    # run_with_statistics(stats,
    #                     lambda: bfgs.nonlinear_regression.BFGS(start, math_func.vectorize(f), g),
    #                     lambda ans: show_bfgs_ans(f, ans[0][-1], ans[1]))
    # print("________________")
    # run_with_statistics(stats,
    #                     lambda: scipy.optimize.minimize(math_func.vectorize(f), start, method="BFGS"),
    #                     lambda ans: print(ans) and show_bfgs_ans(f, ans["x"], ans["nit"]))

    # stats = [
    #     lambda: math_func.g_exponent_apply_count,
    # ]
    #
    # g_fact = math_func.g_exponent
    # correct = 2., 1.
    # for start, bounds in [
    #     (np.array([0., 0.]), None),
    #     (np.array([0., 0.]), scipy.optimize.Bounds(np.array([-5, -5]), np.array([4, 5]), keep_feasible=True)),
    #     (np.array([0., 0.]), scipy.optimize.Bounds(np.array([-5, -5]), np.array([3, 5]), keep_feasible=True)),
    #     (np.array([0., 0.]), scipy.optimize.Bounds(np.array([0, -5]), np.array([2, 1]), keep_feasible=True))
    # ]:
    #     print(f"Start: {start}")
    #     if bounds:
    #         print(f"Bounds: [{bounds.lb[0]}:{bounds.ub[0]}] X [{bounds.lb[1]}:{bounds.ub[1]}]")
    #     else:
    #         print("No bounds")
    #     g_apply, difs = g_fact(*correct)
    #     dots = math_func.generate_dots_on_line(100, g_apply, 0, x_from=0, x_to=2)
    #     bounds_analysis_Powell(dots, g_fact, correct, start, bounds, stats)
    #     print("_" * 40)

    # stats = [
    #     lambda: math_func.g_2sin_apply_count,
    # ]
    #
    # g_fact = math_func.g_2sin
    # correct = 2., 1.
    # for start, bounds in [
    #     (np.array([0., 0.]), None),
    #     (np.array([0., 0.]), scipy.optimize.Bounds(np.array([-5, -5]), np.array([4, 5]), keep_feasible=True)),
    #     (np.array([0., 0.]), scipy.optimize.Bounds(np.array([-5, -5]), np.array([3, 5]), keep_feasible=True)),
    #     (np.array([0., 0.]), scipy.optimize.Bounds(np.array([0, -5]), np.array([2, 1]), keep_feasible=True))
    # ]:
    #     print(f"Start: {start}")
    #     if bounds:
    #         print(f"Bounds: [{bounds.lb[0]}:{bounds.ub[0]}] X [{bounds.lb[1]}:{bounds.ub[1]}]")
    #     else:
    #         print("No bounds")
    #     g_apply, difs = g_fact(*correct)
    #     dots = math_func.generate_dots_on_line(100, g_apply, 0, x_from=-2 * math.pi, x_to=2 * math.pi)
    #     bounds_analysis_Powell(dots, g_fact, correct, start, bounds, stats)
    #     print("_" * 40)

    # stats = [
    #     lambda: bfgs.funcs.rosenbrock_func_apply_cnt
    # ]
    #
    # f, g = bfgs.funcs.rosenbrock()
    # correct = 1, 1
    # for start, bounds in [
    #     (np.array([0., 0.]), None),
    #     (np.array([0., 0.]), scipy.optimize.Bounds(np.array([-5, -5]), np.array([5, 5]), keep_feasible=True)),
    #     (np.array([0., 0.]), scipy.optimize.Bounds(np.array([0, 0]), np.array([1, 1]), keep_feasible=True)),
    #     (np.array([6., 2.]), scipy.optimize.Bounds(np.array([2, 2]), np.array([6, 6]), keep_feasible=True))
    # ]:
    #     print(f"Start: {start}")
    #     if bounds:
    #         print(f"Bounds: [{bounds.lb[0]}:{bounds.ub[0]}] X [{bounds.lb[1]}:{bounds.ub[1]}]")
    #     else:
    #         print("No bounds")
    #     bounds_analysis_BFGS(f, correct, start, bounds, stats)
    #     print("_" * 40)

    # stats = [
    #     lambda: bfgs.funcs.himmelblau_func_apply_cnt
    # ]
    #
    # f, g = bfgs.funcs.himmelblau()
    #
    # for correct, start, bounds in [
    #     ((3, 2), np.array([0., 0.]), None),
    #     ((3, 2), np.array([0., 0.]), scipy.optimize.Bounds(np.array([0, 0]), np.array([5, 5]), keep_feasible=True)),
    #     ((3, 2), np.array([0., 0.]), scipy.optimize.Bounds(np.array([0, 0]), np.array([3, 2]), keep_feasible=True)),
    #     ((-3.779, -3.283), np.array([0., 0.]),
    #      scipy.optimize.Bounds(np.array([-5, -5]), np.array([1, 1]), keep_feasible=True)),
    #     ((-3.779, -3.283), np.array([-1., -1.]),
    #      scipy.optimize.Bounds(np.array([-5, -5]), np.array([1, 1]), keep_feasible=True))
    #
    # ]:
    #     print(f"Start: {start}")
    #     if bounds:
    #         print(f"Bounds: [{bounds.lb[0]}:{bounds.ub[0]}] X [{bounds.lb[1]}:{bounds.ub[1]}]")
    #     else:
    #         print("No bounds")
    #     bounds_analysis_BFGS(f, correct, start, bounds, stats)
    #     print("_" * 40)

    # f, _ = bfgs.funcs.rosenbrock()
    # plt.rcParams['figure.figsize'] = [10, 7]
    # xx, yy = 100, 30000
    # plot.show_2arg_func_slice(math_func.vectorize(f), x_min=-xx, x_max=xx, y_min=-yy, y_max=yy, show=True,
    #                           dots_show=False, contour=True)

    # start = np.array([1, 0])
    # A = [[1, 1], [1, -1], [0, 1]]
    # lb = [1, -1, 0]
    # ub = [3, 1, 1.5]
    # constraints = [
    #     LinearConstraint(A, lb=lb, ub=ub)
    # ]
    # way = [start]
    # run_with_statistics([],
    #                     lambda: scipy.optimize.minimize(math_func.vectorize(f), start, method="trust-constr",
    #                                                     constraints=constraints,
    #                                                     callback=lambda x_i, ans_i: way.append(x_i),
    #                                                     options={"maxiter": 1000000}),
    #                     lambda ans: (print(ans),
    #                                  show_way_graph(math_func.vectorize(f), (1, 1), np.array(way),
    #                                                 constraints=constraints)))

    # f, _ = bfgs.funcs.himmelblau()
    # plt.rcParams['figure.figsize'] = [10, 7]
    # xx, yy = 5, 5
    # plot.show_2arg_func_slice(math_func.vectorize(f), x_min=-xx, x_max=xx, y_min=-yy, y_max=yy, show=True,
    #                           dots_show=False, contour=True)

    # start = np.array([-1, -1])

    # A = [[1, -1], [1, 1]]
    # lb = [-2, -7]
    # ub = [3, 7]
    # # local minimum
    # A = [[1, -1], [1, 1]]
    # lb = [-2, -6]
    # ub = [3, 7]
    # constraints = [
    #     LinearConstraint(A, lb=lb, ub=ub)
    # ]
    # way = [start]
    # run_with_statistics([],
    #        /             lambda: scipy.optimize.minimize(math_func.vectorize(f), start, method="trust-constr",
    #                                                     constraints=constraints,
    #                                                     callback=lambda x_i, ans_i: way.append(x_i),
    #                                                     options={"maxiter": 1000000}),
    #                     lambda ans: (print(ans),
    #                                  show_way_graph(math_func.vectorize(f), (3, 2), np.array(way),
    #                                                 constraints=constraints)))

    # f, _ = bfgs.funcs.rosenbrock()
    # plt.rcParams['figure.figsize'] = [10, 7]
    # xx, yy = 100, 30000
    # plot.show_2arg_func_slice(math_func.vectorize(f), x_min=-xx, x_max=xx, y_min=-yy, y_max=yy, show=True,
    #                           dots_show=False, contour=True)
    #
    # start = np.array([-1, 0])

    # def func(xxx: np.ndarray):
    #     x = xxx / 1.9 - np.array([0.5, 0])
    #     res = x.copy()
    #     for _ in range(100):
    #         res = np.array([res[0] ** 2 - res[1] ** 2 + x[0], 2 * res[0] * res[1] + x[1]])
    #         if np.linalg.norm(res) > 100:
    #             return 100
    #     return np.array([np.linalg.norm(res)])
    #
    # lb = np.array([0])
    # ub = np.array([2])

    # def func(x: np.ndarray):
    #     return x[0] ** 2 - x[1]
    # lb = np.array([-0.5])
    # ub = np.array([1])
    # constraints = [
    #     NonlinearConstraint(func, lb, ub)
    # ]
    # way = [start]
    # run_with_statistics([],
    #                     lambda: scipy.optimize.minimize(math_func.vectorize(f), start, method="trust-constr",
    #                                                     constraints=constraints,
    #                                                     callback=lambda x_i, ans_i: way.append(x_i), tol=1e-2,
    #                                                     options={"maxiter": 100000}),
    #                     lambda ans: (print(ans),
    #                                  show_way_graph(math_func.vectorize(f), (1, 1), np.array(way),
    #                                                 constraints=constraints)))
    pass

if __name__ == "__main__":
    main()
