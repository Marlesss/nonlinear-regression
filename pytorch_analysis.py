import pytorch_tool
import stochastic_gradient_descent.func_utils
import stochastic_gradient_descent.main
from stochastic_gradient_descent.gradient_descent import *
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

PLOT_HEIGHT, PLOT_WIDTH = 3, 5


def custom_sgd(method):
    def sgd(batch_size: int, lr: float, epoch_limit: int):
        if method == 'SGD':
            return stochastic_factory(batch_size, lr, constant_learning_rate(), func_utils.grad_func, default_new_args,
                                      epoch_limit=epoch_limit)
        elif method == 'Momentum':
            return stochastic_factory(batch_size, lr, constant_learning_rate(), func_utils.grad_func,
                                      momentum_mod(0.95),
                                      epoch_limit=epoch_limit)
        elif method == 'Nesterov':
            return stochastic_factory(batch_size, lr, constant_learning_rate(),
                                      *nesterov_mod(0.95, func_utils.grad_func),
                                      epoch_limit=epoch_limit)
        elif method == 'AdaGrad':
            return stochastic_factory(batch_size, lr, constant_learning_rate(), func_utils.grad_func, adagrad_mod(),
                                      epoch_limit=epoch_limit)
        elif method == 'RMSProp':
            return stochastic_factory(batch_size, lr, constant_learning_rate(), func_utils.grad_func, rmsprop_mod(0.95),
                                      epoch_limit=epoch_limit)
        elif method == 'Adam':
            return stochastic_factory(batch_size, lr, constant_learning_rate(), func_utils.grad_func,
                                      adam_mod(0.9, 0.999),
                                      epoch_limit=epoch_limit)
        else:
            raise RuntimeError("Unsupported method")

    return sgd


def generate_dots_on_line(n: int, line: Callable[[float], float], noize: float = 0, x_from=0, x_to=100) -> np.ndarray:
    return np.array([(x, line(x) + (random.random() - 1 / 2) * noize) for x in np.linspace(x_from, x_to, n)])


def run_with_various(dots, values: [int], title_fact: Callable[[int], str], run_sgd):
    n_rows = (len(values) + 1) // 2
    n_cols = (len(values) + n_rows - 1) // n_rows
    plt.rcParams['figure.figsize'] = [PLOT_WIDTH * n_cols, PLOT_HEIGHT * n_rows]
    for i, val in enumerate(values):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        ax.set_title(title_fact(val), y=1.0, pad=-14)
        converged, way = run_sgd(val)
        ans_args = way[-1][::-1]
        plt.plot(dots[:, 0], dots[:, 1], ".")
        stochastic_gradient_descent.main.draw_solution(dots, ans_args, show=False)
    plt.show()


def compare_batch_size(dots, batch_sizes: [int], method: str, lr=1e-6, epoch_limit=1, start=None):
    if start is None:
        start = [0, 0]

    def run_sgd(b_s):
        return pytorch_tool.torch_sgd_linear(torch.tensor(dots, dtype=torch.float32),
                                             b_s,
                                             lr=lr,
                                             start=start,
                                             method=method,
                                             epoch_limit=epoch_limit, dtype=torch.float32)

    run_with_various(dots, batch_sizes, lambda val: f"Batch {val}", run_sgd)


def compare_epoch_limit(dots, epoch_limits: [int], method: str, lr=1e-6, batch_size=1, start=None):
    if start is None:
        start = [0, 0]

    def run_sgd(e_l):
        return pytorch_tool.torch_sgd_linear(torch.tensor(dots, dtype=torch.float32),
                                             batch_size,
                                             lr=lr,
                                             start=start,
                                             method=method,
                                             epoch_limit=e_l, dtype=torch.float32)

    run_with_various(dots, epoch_limits, lambda val: f"Epoch limit {val}", run_sgd)


def main():
    compare_batch_size(generate_dots_on_line(100, lambda x: 3 * x + 5, noize=30), [1, 10, 25, 50, 75, 100], 'Momentum',
                       epoch_limit=1, lr=2 * 1e-7)
    compare_epoch_limit(generate_dots_on_line(100, lambda x: 3 * x + 5, noize=30), [1, 5, 10, 20, 30, 50], 'SGD',
                        batch_size=100, lr=1e-7)

    ("Default", (1, 0, 0), func_utils.grad_func, default_new_args),
    ("Momentum", (1 / 2, 1 / 2, 1 / 2), func_utils.grad_func, momentum_mod(0.95)),
    ("Nesterov", (1 / 2, 0, 1 / 2), *nesterov_mod(0.95, func_utils.grad_func)),
    ("AdaGrad", (0, 1, 1), func_utils.grad_func, adagrad_mod()),
    ("RMSProp", (0, 1, 0), func_utils.grad_func, rmsprop_mod(0.95)),
    ("Adam", (0, 0, 1), func_utils.grad_func, adam_mod(0.9, 0.999))

    # batch_size = 10
    # epoch_limit = 10000
    # lr = 2.98 * 1e-5
    # dots = generate_dots_on_line(100, lambda x: 3 * x + 5, noize=0)
    # start = [4, 3]
    #
    # start_time = time.time()
    # converged1, way1 = pytorch_tool.torch_sgd_linear(torch.tensor(dots, dtype=torch.float32), batch_size, lr=lr,
    #                                                  start=start,
    #                                                  epoch_limit=epoch_limit, dtype=torch.float32)
    # finish_time = time.time()
    # duration1 = finish_time - start_time
    # print(f"Time is {duration1}")
    # print(converged1, way1, len(way1))
    # print(f"Mistake of solution is {stochastic_gradient_descent.main.linear_regression_mistake(dots)(way1[-1][::-1])}")
    #
    # start_time = time.time()
    # converged2, way2 = stochastic_factory(batch_size, lr, constant_learning_rate(), func_utils.grad_func,
    #                                       default_new_args,
    #                                       epoch_limit=epoch_limit, log=0)(torch.tensor(dots), start)
    # finish_time = time.time()
    # duration2 = finish_time - start_time
    # way2 = way2[:, ::-1]
    # print(f"Time is {duration2}")
    # print(converged2, way2, len(way2))
    # print(f"Mistake of solution is {stochastic_gradient_descent.main.linear_regression_mistake(dots)(way2[-1][::-1])}")


if __name__ == "__main__":
    main()
