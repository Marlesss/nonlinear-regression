import numpy as np

from nonlinear_regression import *
from plot import *


def float_range(fr, to, step):
    ans = []
    while fr <= to:
        ans.append(fr)
        fr += step
    return ans


def test1():
    dots = np.array([(i, math.e ** i) for i in float_range(-2, 2, 1 / 4)])
    print(dots)
    ans = gauss_newton(dots, np.array([1 / 10, 1]), g_exponent)
    ans_func = g_exponent(*ans)[0]
    print(ans)
    print(f"Mistake of solution: {mistake(dots, ans_func)}")
    plt.plot(dots[:, 0], dots[:, 1], 'o')
    show_2arg_func(None, np.array([(x, ans_func(x)) for x in np.linspace(min(dots[:, 0]), max(dots[:, 0]), GRID_SIZE)]))


def test2():
    dots = np.array([(i, 1 / 2 * math.e ** (4 * i)) for i in float_range(-2, 2, 1 / 4)])
    plt.plot(dots[:, 0], dots[:, 1], 'o')
    plt.show()
    ans = gauss_newton(dots, np.array([1, 1]), g_exponent)
    print(ans)
    print(f"Mistake of solution: {mistake(dots, g_exponent(*ans)[0])}")


def test3():
    dots = np.array([(i, math.sin(5 * i)) for i in float_range(-1, 1, 1 / 20)])
    plt.plot(dots[:, 0], dots[:, 1], 'o', color=(1, 0, 0))
    ans = gauss_newton(dots, np.array([1]), g_sin)
    print(ans)
    print(f"Mistake of solution: {mistake(dots, g_sin(*ans)[0])}")

    x_space = np.linspace(-1, 1, GRID_SIZE)
    show_2arg_func(None, np.array([[x, g_sin(5)[0](x)] for x in x_space]), show=False, color=(0, 0, 1))
    show_2arg_func(None, np.array([[x, g_sin(*ans)[0](x)] for x in x_space]), color=(0, 1, 0))


def test4(correct):
    # correct = [4.69669771852408, 0.13432756708851563]
    start = [3.0, 0.1]
    g_factory = g_2sin
    func, _ = g_factory(*correct)
    dots = np.array([(i, func(i)) for i in np.linspace(0.5, 20, 50)])
    plt.plot(dots[:, 0], dots[:, 1], 'o', color=(1, 0, 0))
    ans = gauss_newton(dots, np.array(start), g_factory)
    print(ans)
    print(f"Mistake of solution: {mistake(dots, g_factory(*ans)[0])}")

    x_space = np.linspace(0.5, 20, GRID_SIZE)
    show_2arg_func(None, np.array([[x, g_factory(*correct)[0](x)] for x in x_space]), show=False, color=(0, 0, 1))
    show_2arg_func(None, np.array([[x, g_factory(*ans)[0](x)] for x in x_space]), color=(0, 1, 0))


def main():
    print("nonlinear regression")
    test1()
    # test2()
    # test3()
    # test4([4.69669771852408, 0.13432756708851563])
    # test4([3.0595462134077707, 0.18291230660078392])


if __name__ == "__main__":
    main()
