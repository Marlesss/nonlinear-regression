import pytorch_tool
import stochastic_gradient_descent.func_utils
import stochastic_gradient_descent.main
from math_func import generate_dots_on_line
from stochastic_gradient_descent.gradient_descent import *
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import time

PLOT_HEIGHT, PLOT_WIDTH = 3, 5


def custom_sgd(method):
    def sgd(batch_size: int, lr: float, epoch_limit: int):
        if method == 'SGD':
            return stochastic_factory(batch_size, lr, constant_learning_rate(), func_utils.grad_func, default_new_args,
                                      epoch_limit=epoch_limit)
        elif method == 'Momentum':
            return stochastic_factory(batch_size, lr, constant_learning_rate(), func_utils.grad_func,
                                      momentum_mod(0.9),
                                      epoch_limit=epoch_limit)
        elif method == 'Nesterov':
            return stochastic_factory(batch_size, lr, constant_learning_rate(),
                                      *nesterov_mod(0.95, func_utils.grad_func),
                                      epoch_limit=epoch_limit)
        elif method == 'AdaGrad':
            return stochastic_factory(batch_size, lr, constant_learning_rate(), func_utils.grad_func, adagrad_mod(),
                                      epoch_limit=epoch_limit)
        elif method == 'RMSProp':
            return stochastic_factory(batch_size, lr, constant_learning_rate(), func_utils.grad_func, rmsprop_mod(0.9),
                                      epoch_limit=epoch_limit)
        elif method == 'Adam':
            return stochastic_factory(batch_size, lr, constant_learning_rate(), func_utils.grad_func,
                                      adam_mod(0.9, 0.999),
                                      epoch_limit=epoch_limit)
        else:
            raise RuntimeError("Unsupported method")

    return sgd


def run_with_various(dots, values: [int], title_fact: Callable[[int], str], run_sgd):
    n_rows = (len(values) + 1) // 2
    n_cols = (len(values) + n_rows - 1) // n_rows
    plt.rcParams['figure.figsize'] = [PLOT_WIDTH * n_cols, PLOT_HEIGHT * n_rows]
    for i, val in enumerate(values):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.set_title(title_fact(val), y=1.0, pad=-14)
        converged, way = run_sgd(val)
        print(way)
        ans_args = way[-1][::-1]
        plt.plot(dots[:, 0], dots[:, 1], ".")
        stochastic_gradient_descent.main.draw_solution(dots, ans_args, show=False)
    plt.show()


def compare_batch_size(dots, batch_sizes: [int], method: str, lr=1e-6, epoch_limit=1, start=None):
    if start is None:
        start = [0, 0]

    def run_sgd(b_s):
        return pytorch_tool.torch_sgd_linear(torch.tensor(dots, dtype=torch.float64),
                                             b_s,
                                             lr=lr,
                                             start=start,
                                             method=method,
                                             epoch_limit=epoch_limit, dtype=torch.float64, log=True)

    run_with_various(dots, batch_sizes, lambda val: f"Batch {val}", run_sgd)


def compare_epoch_limit(dots, epoch_limits: [int], method: str, lr=1e-6, batch_size=1, start=None):
    if start is None:
        start = [0, 0]

    def run_sgd(e_l):
        return pytorch_tool.torch_sgd_linear(torch.tensor(dots, dtype=torch.float64),
                                             batch_size,
                                             lr=lr,
                                             start=start,
                                             method=method,
                                             epoch_limit=e_l, dtype=torch.float64)

    run_with_various(dots, epoch_limits, lambda val: f"Epoch limit {val}", run_sgd)


def compare_method(method: str, cnt: int):
    def run(dots: np.ndarray, batch_size, start=None, lr=1e-6, epoch_limit=100):
        start_time = time.time()
        converged1, way1 = custom_sgd(method)(batch_size, lr, epoch_limit)(dots, start)
        end_time = time.time()
        time1 = end_time - start_time
        ans1 = way1[-1]
        mistake1 = stochastic_gradient_descent.main.linear_regression_mistake(dots)(ans1)
        print(f"way1 length = {len(way1)}; ans = {ans1}; time = {time1}; mistake = {mistake1}")

        start_time = time.time()
        converged2, way2 = pytorch_tool.torch_sgd_linear(torch.tensor(dots), batch_size, start, lr, epoch_limit, method)
        end_time = time.time()
        time2 = end_time - start_time
        ans2 = way2[-1][::-1]
        mistake2 = stochastic_gradient_descent.main.linear_regression_mistake(dots)(ans2)
        print(f"way2 length = {len(way2)}; ans = {ans2}; time = {time2}; mistake = {mistake2}")
        return [len(way1), time1], [len(way2), time2]

    def average(dots_count=100, batch_size=100, start=None, lr=1e-6, epoch_limit=100):
        sum_r1, sum_t1, sum_r2, sum_t2 = 0, 0, 0, 0
        for _ in range(cnt):
            a = random.randint(1, 5)
            b = random.randint(-10, 10)
            noize = random.randint(0, 0)
            dots = generate_dots_on_line(dots_count, lambda x: a * x + b, noize)
            (r1, t1), (r2, t2) = run(dots, batch_size, start, lr, epoch_limit)
            sum_r1 += r1
            sum_t1 += t1
            sum_r2 += r2
            sum_t2 += t2
            print(f"y = {a} * x + {b} with noize {noize}")
            print(f"Results: ({r1}, {t1}), ({r2}, {t2})")
            print(f"___________________")
        print(f"Average results: ({sum_r1 / cnt}, {sum_t1 / cnt}), ({sum_r2 / cnt}, {sum_t2 / cnt})")

    return average


def main():
    # compare_batch_size(generate_dots_on_line(100, lambda x: 3 * x + 5, noize=30), [1, 10, 25, 50, 75, 100], 'AdaGrad',
    #                    epoch_limit=1, lr=2)
    # compare_epoch_limit(generate_dots_on_line(100, lambda x: 3 * x + 5, noize=30), [1, 5, 10, 20, 30, 50], 'SGD',
    #                     batch_size=100, lr=1e-7)
    #
    # for dots_count, batch_size in [
    #     (100, 100),
    #     (100, 50)]:
    #     print(f"dots: {dots_count}, batch_size: {batch_size}")
    #     compare_method("Adam", 10)(dots_count, batch_size, [0, 0], 0.005, 10000)
    pass


if __name__ == "__main__":
    main()
