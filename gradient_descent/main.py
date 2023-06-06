import random

import numpy as np
from matplotlib.patches import Patch

from gradient_descent import *
from square_function import generate_function
from plot import *
from pprint import pprint
import math


def get_short_func():
    def ploho(x: np.ndarray):
        return x[0] ** 2

    def ploho_grad(x: np.ndarray):
        return np.array([2 * x[0], 0])

    return f"f(x_1, x_2) = x_1^2", ploho, ploho_grad


def get_simple_func(k: float):
    def ploho(x: np.ndarray) -> float:
        return k * x[0] ** 2 + x[1] ** 2

    def ploho_grad(x: np.ndarray) -> np.ndarray:
        return np.array([2 * k * x[0], 2 * x[1]])

    return f"f(x_1, x_2) = {k} * x_1^2 + x_2^2", ploho, ploho_grad


def get_gradient_descent_constant(alph: float):
    def wrapped(func: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], start_dot: np.ndarray):
        return gradient_descent_constant(start_dot, alph, grad)

    return f"градиентного спуска с постоянным шагом ({alph})", wrapped


def get_gradient_descent_linear():
    def wrapped(func: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], start_dot: np.ndarray):
        return gradient_descent_linear(start_dot, grad, func)

    return "гр. с. с одномерным поиском (м. золотого сечения)", wrapped


def get_gradient_descent_linear_with_wolfe_condition(c1: float, c2: float):
    def wrapped(func: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray], start_dot: np.ndarray):
        return gradient_descent_linear_with_wolfe_condition(start_dot, grad, func, c1, c2)

    return "гр. с. с одномерным поиском (м. золотого сечения с условием Вольфе)", wrapped


def call_grad_descent_in_dot(func, grad, gradient_descent, start_dot: np.ndarray):
    def calc_func_usage(func):
        def wrapped(*args, **kwargs):
            wrapped.usage += 1
            return func(*args, **kwargs)

        wrapped.usage = 0
        return wrapped

    func = calc_func_usage(func)
    grad = calc_func_usage(grad)
    conv, dots = gradient_descent(func, grad, start_dot)
    return conv, dots, func.usage, grad.usage


def scale_func(scale: np.ndarray, func: Callable[[np.ndarray], float], grad: Callable[[np.ndarray], np.ndarray]):
    def wrapped_func(x: np.ndarray) -> float:
        return func(np.dot(x, np.diag(scale)))

    def wrapped_grad(x: np.ndarray) -> np.ndarray:
        return grad(np.dot(np.dot(x, np.diag(scale)), np.diag(scale)))

    return wrapped_func, wrapped_grad


def test_4():
    print("4 Пункт")
    print("=========================")
    i = 0
    for func_name, func, grad in [
        get_simple_func(1),
        get_simple_func(10),
        get_short_func()
    ]:
        i += 1
        for gradient_descent_name, gradient_descent in [
            get_gradient_descent_constant(0.01),
            get_gradient_descent_constant(0.05),
            get_gradient_descent_constant(0.3),
            get_gradient_descent_linear(),
            get_gradient_descent_linear_with_wolfe_condition(10 ** (-6), 2 * 10 ** (-6))
        ]:
            print("=========================")
            print(func_name)
            print(gradient_descent_name)
            print("_________________________")

            x_min, x_max, y_min, y_max = -100, 100, -100, 100
            x_space = np.linspace(x_min, x_max, x_max - x_min + 1)
            y_space = np.linspace(y_min, y_max, x_max - x_min + 1)
            grad_descent_in_dot = [[call_grad_descent_in_dot(func, grad, gradient_descent, np.array([x, y]))
                                    for x in x_space]
                                   for y in y_space]

            print("Сходимость относительно точки старта")
            plt.title(f"Сходимость относительно точки старта\nдля градиентного спуска")
            convergence = [[grad_descent_in_dot[iy][ix][0] for ix in range(len(x_space))] for iy in range(len(y_space))]
            plt.contourf(x_space, y_space, convergence)
            blue_patch = Patch(color='royalblue', label='Мн-во точек, в которых гр. с. не сошёлся')
            green_patch = Patch(color='limegreen', label='Мн-во точек, в которых гр. с. сошёлся')
            plt.legend(handles=[blue_patch, green_patch], loc='upper right', fontsize="xx-small")
            plt.show()

            print("Количество вызовов функции и ее градиента")
            plt.title(f"Количество вызовов функции и ее градиента\nдля {gradient_descent_name} для f{i}")
            func_grad_usage = [[grad_descent_in_dot[iy][ix][2] + 2 * grad_descent_in_dot[iy][ix][3]
                                for ix in range(len(x_space))]
                               for iy in range(len(y_space))]
            cs1 = plt.contourf(x_space, y_space, func_grad_usage)
            cs2 = plt.contour(x_space, y_space, func_grad_usage)
            cbar = plt.colorbar(cs1)
            cbar.add_lines(cs2)
            plt.legend()
            plt.show()

            x, y = 0, 20
            print(f"Градиентный спуск из точки ({x}, {y})")
            conv, dots = gradient_descent(func, grad, np.array([x, y]))
            plt.title(f"Градиентный спуск\nиз точки ({x}, {y})")
            show_2arg_func(func, dots, levels=False, contour=True, label="Траектория градиентного спуска")

            for r in [5, 20, 50]:
                cnt = 8
                all_dots = None
                print(
                    f"Шаги градиентного спуска относительно точки старта ({cnt} точек старта на расстоянии {r} от минимума)")
                plt.title(
                    f"Шаги градиентного спуска относительно\nточки старта ({cnt} точек старта на расстоянии {r} от минимума)")
                for i in range(cnt):
                    color = (i / cnt, 1 - i / cnt, 0)
                    x = r * math.cos(math.pi * i / cnt * 2)
                    y = r * math.sin(math.pi * i / cnt * 2)
                    _, dots = gradient_descent(func, grad, np.array([x, y]))
                    all_dots = np.concatenate((all_dots, dots)) if all_dots is not None else dots
                    show_2arg_func(func, dots, show=False, color=color,
                                   label=f"Траектория гр. с. из т. ({'%.2f' % x}, {'%.2f' % y})")
                show_2arg_func(func, all_dots, dots_show=False, levels=False, contour=True)
            print("=========================")


def test_4d():
    print("Пункт 4d")
    func_name, func, grad = get_simple_func(10)
    scale = np.array([1 / 10 ** (1 / 2), 1])
    scaled_func, scaled_grad = scale_func(scale, func, grad)
    descent_name, descent = get_gradient_descent_constant(0.3)
    print(f"Функция {func_name}")
    print(f"Нормализованная функция: f'(x_1, x_2) = x_1^2 + x_2^2")
    start_dot = np.array([20, 20])
    _, dots = descent(func, grad, start_dot)
    _, scaled_dots = descent(scaled_func, scaled_grad, start_dot)
    plt.title(f"Градиентный спуск для\nисходной функции f2(x, y) = 10x^2 + y^2")
    show_2arg_func(func, dots, levels=False, contour=True, label="Траектория градиентного спуска")
    plt.title(f"Градиентный спуск для\nнормализованной функции f2'(x, y) = x^2 + y^2")
    show_2arg_func(scaled_func, scaled_dots, levels=False, contour=True, label="Траектория градиентного спуска")
    print(f"Минимум нормализованной функции в точке {scaled_dots[-1]}")
    print(f"Минимум в исходной функции в точке {np.dot(scaled_dots[-1], np.diag(1 / scale))}")


def test_6(gradient_descent_name, gradient_descent):
    n_grid_size = 20
    k_grid_size = 20
    repeat_count = 5
    random_dots = 5
    n_space = np.linspace(2, 1000, n_grid_size)
    k_space = np.linspace(1, 1000, k_grid_size)
    data = [[None for _ in range(k_grid_size)] for _ in range(n_grid_size)]
    for i_n in range(n_grid_size):
        print(i_n)
        n = int(n_space[i_n])
        for i_k in range(k_grid_size):
            k = int(k_space[i_k])
            avg_convergence = 0
            for _ in range(repeat_count):
                f, grad = generate_function(n, k)
                for _ in range(random_dots):
                    start_dot = [random.randint(-100, 100) for _ in range(n)]
                    conv, dots = gradient_descent(f, grad, np.array(start_dot))
                    avg_convergence += len(dots)
            avg_convergence /= repeat_count * random_dots
            data[i_n][i_k] = avg_convergence
    print("Количество итераций, необходимых для сходимости")
    plt.title(f"Количество итераций, необходимых для сходимости\n{gradient_descent_name}")
    cs1 = plt.contourf(n_space, k_space, data)
    cs2 = plt.contour(n_space, k_space, data)
    cbar = plt.colorbar(cs1)
    cbar.add_lines(cs2)
    plt.xlabel("n")
    plt.ylabel("k")
    plt.show()


def main():
    test_4()
    test_4d()
    # test_6(*get_gradient_descent_linear())


if __name__ == '__main__':
    main()
