import time

import numpy as np

from plot import *
from stochastic_gradient_descent.gradient_descent import *


def linear_regression_mistake(dots: np.ndarray) -> Callable[[np.ndarray], float]:
    return lambda func_arg: func_utils.mistake(func_utils.func(func_utils.curve(func_arg)), dots)


def add_solution_to_plot(dots: np.ndarray, way: np.ndarray, show=True, color=(1, 0, 0), label: str = "solution"):
    ans = way[-1]
    draw_solution(dots, ans, show=show, color=color, label=label)


def draw_solution(dots: np.ndarray, ans_args: np.ndarray, show=True, color=(1, 0, 0), label: str = "solution"):
    curve = func_utils.curve(ans_args)
    x_space = np.linspace(min(dots[:, 0]), max(dots[:, 0]), 30)
    plt.plot(x_space, [curve(x) for x in x_space], color=color, label=label)
    if show:
        plt.show()


def add_way_to_plot(dots: np.ndarray, way: np.ndarray, **kwargs):
    show_2arg_func(linear_regression_mistake(dots), way, **kwargs)


def show_ways_contour(dots, ways: [np.ndarray]):
    show_2arg_func_slice(linear_regression_mistake(dots),
                         x_min=min(map(lambda way: min(way[:, 0]), ways)),
                         x_max=max(map(lambda way: max(way[:, 0]), ways)),
                         y_min=min(map(lambda way: min(way[:, 1]), ways)),
                         y_max=max(map(lambda way: max(way[:, 1]), ways)),
                         show=False, dots_show=False, contour=True)


def main():
    scattered = np.array([
        (1, 2),
        (2, 3),
        (0, 3),
        (-1, 1),
        (-2, 0),
        (4, 5),
        (-4, 0),
        (-5, -2),
        (-3, -1),
        (3, 6),
        (-6, -1),
        (5, 6)
    ])
    test1(scattered)
    # test1(np.array([(x, random.randint(-10, 10))
    #                 for x in range(-6, 6)
    #                 ]))
    # test1(np.array([(x, 3.75 * x + 4.5) for x in range(-6, 6)]))
    test2a(scattered)
    test2b(scattered)
    test3(scattered)
    test4(scattered)
    # test4(np.array([(x, 3.75 * x + 4.5) for x in range(-6, 6)]))
    test5(scattered)

    test6(np.array([(x, math.sin(x)) for x in range(-6, 6)]))
    somefunc = np.array([[-2., 0.13533528],
                         [-1.75, 0.17377394],
                         [-1.5, 0.22313016],
                         [-1.25, 0.2865048],
                         [-1., 0.36787944],
                         [-0.75, 0.47236655],
                         [-0.5, 0.60653066],
                         [-0.25, 0.77880078],
                         [0., 1.],
                         [0.25, 1.28402542],
                         [0.5, 1.64872127],
                         [0.75, 2.11700002],
                         [1., 2.71828183],
                         [1.25, 3.49034296],
                         [1.5, 4.48168907],
                         [1.75, 5.75460268],
                         [2., 7.3890561]])
    test7(somefunc)


def test1(dots):
    batch_sizes = [1, 2, 3, 4, 6, 12]
    # learning_rate = [0.01, 0.001]
    stochastic = lambda batch_size: stochastic_factory(batch_size, 0.001, constant_learning_rate(),
                                                       func_utils.grad_func,
                                                       default_new_args, epoch_limit=2000)(dots,
                                                                                           np.array([1.0, 0.0]))
    ways = [stochastic(bs)[1] for bs in batch_sizes]
    show_ways_contour(dots, ways)
    for i in range(len(batch_sizes)):
        color = (i / len(batch_sizes), 1 - i / len(batch_sizes), 0)
        add_way_to_plot(dots, ways[i], show=False, label=f"batch_size = {batch_sizes[i]}", color=color)
    plt.legend(fontsize="xx-small", loc='upper right')
    plt.show()
    for i in range(len(batch_sizes)):
        color = (i / len(batch_sizes), 1 - i / len(batch_sizes), 0)
        add_solution_to_plot(dots, ways[i], show=False, color=color,
                             label=f"batch_size = {batch_sizes[i]}. Loss: {linear_regression_mistake(dots)(ways[i][-1])}")
    plt.legend(fontsize="xx-small", loc='upper right')
    plt.show()


def test2a(dots):
    coefs = [0.3, 0.5, 0.7, 0.9]
    k = len(coefs)
    ways = [stochastic_factory(3, 0.01, linear_learning_rate_scheduler(coef, 10), func_utils.grad_func,
                               default_new_args, epoch_limit=2000)(dots, np.array([1.0, 0.0]))[1] for coef in coefs]
    show_ways_contour(dots, ways)
    for i in range(k):
        color = (i / k, 1 - i / k, 0)
        add_way_to_plot(dots, ways[i], show=False, label=f"coef = {coefs[i]}", color=color)
    plt.legend(fontsize="xx-small", loc='upper right')
    plt.show()
    for i in range(k):
        color = (i / k, 1 - i / k, 0)
        add_solution_to_plot(dots, ways[i], show=False, color=color,
                             label=f"coef = {coefs[i]}. Loss: {linear_regression_mistake(dots)(ways[i][-1])}")
    plt.legend(fontsize="xx-small", loc='upper right')
    plt.show()


def test2b(dots):
    coefs = [0.2, 0.5, 0.7, 0.9]
    k = len(coefs)
    ways = [stochastic_factory(3, 0.03, exponent_learning_rate_scheduler(coef), func_utils.grad_func,
                               default_new_args, epoch_limit=2000)(dots, np.array([1.0, 0.0]))[1] for coef in coefs]
    show_ways_contour(dots, ways)
    for i in range(k):
        color = (i / k, 1 - i / k, 0)
        add_way_to_plot(dots, ways[i], show=False, label=f"coef = {coefs[i]}", color=color)
    plt.legend(fontsize="xx-small", loc='upper right')
    plt.show()
    for i in range(k):
        color = (i / k, 1 - i / k, 0)
        add_solution_to_plot(dots, ways[i], show=False, color=color,
                             label=f"coef = {coefs[i]}. Loss: {linear_regression_mistake(dots)(ways[i][-1])}")
    plt.legend(fontsize="xx-small", loc='upper right')
    plt.show()


def test3(dots):
    _, default_way = stochastic_factory(2, 0.005, constant_learning_rate(), func_utils.grad_func,
                                        default_new_args)(dots, np.array([1.0, 0.0]))
    ways = [(mod_name, stochastic_factory(2, ilr, constant_learning_rate(),
                                          grad_func, new_args_func)(dots, np.array([1.0, 0.0])))
            for mod_name, ilr, grad_func, new_args_func in [
                ("Momentum", 0.01, func_utils.grad_func, momentum_mod(0.95)),
                ("Nesterov", 0.01, *nesterov_mod(0.95, func_utils.grad_func)),
                ("AdaGrad", 0.2, func_utils.grad_func, adagrad_mod()),
                ("RMSProp", 0.01, func_utils.grad_func, rmsprop_mod(0.95)),
                ("Adam", 0.03, func_utils.grad_func, adam_mod(0.9, 0.999))
            ]]
    for mod_name, (converges, way) in ways:
        add_way_to_plot(dots, default_way,
                        label=f"Default. Loss: {linear_regression_mistake(dots)(default_way[-1])}",
                        color=(0, 1, 0), show=False)
        add_way_to_plot(dots, way, label=f"{mod_name}. Loss: {linear_regression_mistake(dots)(way[-1])}",
                        color=(1, 0, 0), show=False)
        plt.legend(fontsize="xx-small", loc='upper right')
        plt.show()


def test4(dots):
    prev_func_usage = func_utils.FUNC_USAGE
    prev_grad_usage = func_utils.GRAD_USAGE
    for mod_name, color, ilr, grad_func, new_args_func in [
        ("Default", (1, 0, 0), 0.002, func_utils.grad_func, default_new_args),
        ("Momentum", (1 / 2, 1 / 2, 1 / 2), 0.02, func_utils.grad_func, momentum_mod(0.95)),
        ("Nesterov", (1 / 2, 0, 1 / 2), 0.004, *nesterov_mod(0.95, func_utils.grad_func)),
        ("AdaGrad", (0, 1, 1), 0.8, func_utils.grad_func, adagrad_mod()),
        ("RMSProp", (0, 1, 0), 0.08, func_utils.grad_func, rmsprop_mod(0.9)),
        ("Adam", (0, 0, 1), 0.12, func_utils.grad_func, adam_mod(0.9, 0.999))
        # ("Default", (1, 0, 0), 0.0005, func_utils.grad_func, default_new_args),
        # ("Momentum", (1 / 2, 1 / 2, 1 / 2), 0.005, func_utils.grad_func, momentum(0.95)),
        # ("Nesterov", (1 / 2, 0, 1 / 2), 0.001, *nesterov_mod(0.95, func_utils.grad_func)),
        # ("AdaGrad", (0, 1, 1), 0.2, func_utils.grad_func, adagrad_mod()),
        # ("RMSProp", (0, 1, 0), 0.02, func_utils.grad_func, rmsprop_mod(0.9)),
        # ("Adam", (0, 0, 1), 0.03, func_utils.grad_func, adam_mod(0.9, 0.999))
    ]:
        start_time = time.time()
        conv, way = stochastic_factory(4, ilr, constant_learning_rate(), grad_func, new_args_func,
                                       epoch_limit=4000)(dots, np.array([1.0, 0.0]))
        work_time = time.time() - start_time
        print("________________")
        print(mod_name)
        print(conv)
        print(len(way))
        print(f"FUNC_USAGE: {func_utils.FUNC_USAGE - prev_func_usage}")
        print(f"GRAD_USAGE: {func_utils.GRAD_USAGE - prev_grad_usage}")
        print(f"WORK_TIME: {work_time}")
        print(linear_regression_mistake(dots)(way[-1]))
        print("________________")
        prev_func_usage = func_utils.FUNC_USAGE
        prev_grad_usage = func_utils.GRAD_USAGE


def test5(dots):
    ways = [(mod_name, color, stochastic_factory(2, 0.01, constant_learning_rate(),
                                                 grad_func, new_args_func)(dots, np.array([1.0, 0.0])))
            for mod_name, color, grad_func, new_args_func in [
                ("Default", (1, 0, 0), func_utils.grad_func, default_new_args),
                ("Momentum", (1 / 2, 1 / 2, 1 / 2), func_utils.grad_func, momentum_mod(0.95)),
                ("Nesterov", (1 / 2, 0, 1 / 2), *nesterov_mod(0.95, func_utils.grad_func)),
                ("AdaGrad", (0, 1, 1), func_utils.grad_func, adagrad_mod()),
                ("RMSProp", (0, 1, 0), func_utils.grad_func, rmsprop_mod(0.95)),
                ("Adam", (0, 0, 1), func_utils.grad_func, adam_mod(0.9, 0.999))
            ]]
    show_ways_contour(dots, [way for mod_name, color, (converges, way) in ways])
    for mod_name, color, (converges, way) in ways:
        add_way_to_plot(dots, way, label=f"{mod_name}. Loss: {linear_regression_mistake(dots)(way[-1])}",
                        color=color, show=False)
    plt.legend(fontsize="xx-small", loc='upper right')
    plt.show()


def test6(dots):
    # way = polynomial_stochastic_gradient_descent(dots, 4, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.00000001, epoch_limit=1000000)
    conv, way = stochastic_factory(4, 0.000002, exponent_learning_rate_scheduler(0.999), func_utils.grad_func,
                                   default_new_args,
                                   epoch_limit=20000)(dots, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
    print(conv)
    print(linear_regression_mistake(dots)(way[-1]))
    add_solution_to_plot(dots, way)


def test7(dots):
    way1 = polynomial_stochastic_gradient_descent(dots, 7, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.00001,
                                                  epoch_limit=1000)
    way2 = polynomial_stochastic_gradient_descent(dots, 7, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.00001,
                                                  epoch_limit=1000, l1=10)
    way3 = polynomial_stochastic_gradient_descent(dots, 7, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.00001,
                                                  epoch_limit=1000, l2=-10)
    way4 = polynomial_stochastic_gradient_descent(dots, 7, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 0.00001,
                                                  epoch_limit=1000, l1=3, l2=5, elastic=1)
    add_solution_to_plot(dots, way1, show=False, color=(1, 0, 0), label="default")
    add_solution_to_plot(dots, way2, show=False, color=(0, 1, 0), label="l1_regu(10)")
    add_solution_to_plot(dots, way3, show=False, color=(0, 0, 1), label="l2_regu(-10)")
    add_solution_to_plot(dots, way4, show=False, color=(0, 0, 0), label="elastic(1)")

    plt.legend(fontsize="xx-small", loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
