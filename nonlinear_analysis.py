import numpy as np
import math
from plot import *
import math_func
from random import random
from gauss_newton.nonlinear_regression import gauss_newton
from powell_dog_leg.nonlinear_regression import powell_dog_leg
from stochastic_gradient_descent.gradient_descent import *
from stochastic_gradient_descent.func_utils import *
from stochastic_gradient_descent.main import add_solution_to_plot
from bfgs.nonlinear_regression import BFGS, L_BFGS
from gradient_descent.gradient_descent import gradient_descent_constant, gradient_descent_linear
import bfgs.funcs

adam_gd = lambda lr: stochastic_factory(10, lr, constant_learning_rate(), grad_func, adam_mod(0.9, 0.999), epoch_limit=50)


def add_gd_solution_to_plot(ans_func, dots, label, color):
    print(f"Mistake of solution {label}: {math_func.mistake(dots, ans_func)}")
    show_2arg_func(None, np.array([(x, ans_func(x)) for x in np.linspace(min(dots[:, 0]), max(dots[:, 0]), GRID_SIZE)]),
                   label=label, color=color, show=False)


def addStochasticLinear(dots: np.ndarray, g_2arg, start: np.ndarray, adam_lr, polynomial_dim, polynomial_lr):
    _, way = adam_gd(adam_lr)(dots, start)

    # show_2arg_func(lambda args: math_func.mistake(dots, g_2arg(*args)[0]), way, contour=True, quiver=True)
    # show_2arg_func(lambda args: math_func.mistake(dots, g_2arg(*args)[0]), way,
    #                x_space=np.linspace(0, 2.5, GRID_SIZE), y_space=np.linspace(-1, 3, GRID_SIZE), contour=True, quiver=True)

    ans = way[-1]
    ans_func = g_2arg(*ans)[0]
    add_gd_solution_to_plot(ans_func, dots, label="Adam", color=(0, 1, 0))

    way = polynomial_stochastic_gradient_descent(dots, 7, np.array([0.0 for i in range(polynomial_dim)]), polynomial_lr,
                                                 epoch_limit=50)
    add_solution_to_plot(dots, way, show=False, color=(0, 0, 1), label="Polynomial")


def testNewtonAndDog(dots, g_2arg, start, adam_lr, polynomial_dim, polynomial_lr):
    addStochasticLinear(dots, g_2arg, start, adam_lr, polynomial_dim, polynomial_lr)

    ans = gauss_newton(dots, start, g_2arg)
    print(ans)
    ans_func = g_2arg(*ans)[0]
    add_gd_solution_to_plot(ans_func, dots, label="Gauss-Newton", color=(1, 0, 0))

    ans = powell_dog_leg(dots, start, g_2arg)
    ans_func = g_2arg(*ans)[0]
    add_gd_solution_to_plot(ans_func, dots, label="Powell Dog Leg", color=(1, 1, 0))

    plt.plot(dots[:, 0], dots[:, 1], "o")

    plt.legend(fontsize="xx-small", loc='upper right')
    plt.show()

    # t = 2.5
    # show_2arg_func_slice(lambda args: math_func.mistake(dots, g_2arg(*args)[0]), x_min=-t, x_max=t, y_min=-t, y_max=t, dots_show=False, contour=True)
    #
    # t = 10
    # show_2arg_func_slice(lambda args: math_func.mistake(dots, g_2arg(*args)[0]), x_min=-t, x_max=t, y_min=-t, y_max=t, dots_show=False, contour=True)

def get_dots(func, shaking, x_from, x_to, cnt):
    return np.array([(i, func(i) + (random() - 1 / 2) * shaking) for i in np.linspace(x_from, x_to, cnt)])


def runTestGaussNewtonLeg():
    dots = get_dots(math_func.g_2sin(1.0, 1.5)[0], 0.4, -2, 2, 20)
    testNewtonAndDog(dots, math_func.g_2sin, np.array([2, 1]), adam_lr=0.002, polynomial_dim=6, polynomial_lr=0.00020)

    dots = get_dots(math_func.g_exponent(1.5, 2)[0], 3, -2, 1.5, 20)
    testNewtonAndDog(dots, math_func.g_exponent, np.array([1, 1]), adam_lr=0.005, polynomial_dim=7, polynomial_lr=0.0001)

    dots = get_dots(math_func.g_2parabola(1.4, 4)[0], 2, -2, 2, 20)
    testNewtonAndDog(dots, math_func.g_2parabola, np.array([1, 1]), adam_lr=0.5, polynomial_dim=3, polynomial_lr=0.0005)


def addGDMinimisation(func, grad, start: np.ndarray, constant_lr):
    fufunc = lambda args: func(*args)

    _, way = gradient_descent_constant(start, constant_lr, grad)
    show_2arg_func(fufunc, way, label="constant", color=(0, 1, 0), show=False)

    _, way = gradient_descent_linear(start, grad, fufunc)
    show_2arg_func(fufunc, way, label="golden section", color=(0, 0, 1), show=False)


def testBFGS(func, grad, start, constant_lr):
    fufunc = lambda args: func(*args)

    addGDMinimisation(func, grad, start, constant_lr)

    way = BFGS(start, fufunc, grad)
    show_2arg_func(fufunc, way, label="BFGS", color=(1, 0, 0), show=False)

    way = L_BFGS(start, fufunc, grad)
    show_2arg_func(fufunc, way, label="L_BFGS (10)", color=(1, 1, 0), contour=True)


def testLBFGS(func, grad, start):
    fufunc = lambda args: func(*args)

    ways = []
    for (color, m) in [((1, 0, 0), 1), ((0, 1, 0), 2), ((0, 0, 1), 5), ((1, 1, 0), 10)]:
        way = L_BFGS(start, fufunc, grad, m=m)
        show_2arg_func(fufunc, way, label=f"L_BFGS ({m})", color=color, show=False)
        ways.append(way)
    show_ways_contour(fufunc, ways)
    plt.legend(fontsize="xx-small", loc='upper right')
    plt.show()


def runTestBFGS():
    func, grad = bfgs.funcs.norm()
    testBFGS(func, grad, np.array([2, 1]), constant_lr=0.07)

    func, grad = bfgs.funcs.bad()
    testBFGS(func, grad, np.array([4, -19]), constant_lr=0.07)

    func, grad = bfgs.funcs.bad()
    testLBFGS(func, grad, np.array([4, -19]))


def main():
    runTestGaussNewtonLeg()
    runTestBFGS()


if __name__ == "__main__":
    main()
