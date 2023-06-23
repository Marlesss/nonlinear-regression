from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import LinearConstraint, NonlinearConstraint

GRID_SIZE = 100


def show_2arg_func(f: Callable[[np.ndarray], float], dots: np.ndarray, dots_show: bool = True, levels: bool = False,
                   contour: bool = False, show=True, label: str = "dots", clabel: bool = False,
                   color: tuple = (1, 0, 0), x_space=None, y_space=None, quiver=False, width=0.003, alpha=0.7,
                   constraints=None):
    if x_space is None:
        x_min, x_max = min(dots[:, 0]), max(dots[:, 0])
        x_space = np.linspace(x_min - (x_max - x_min) / 10, x_max + (x_max - x_min) / 10, GRID_SIZE)
    if y_space is None:
        y_min, y_max = min(dots[:, 1]), max(dots[:, 1])
        y_space = np.linspace(y_min - (y_max - y_min) / 10, y_max + (y_max - y_min) / 10, GRID_SIZE)
    if levels:
        contour_set = plt.contour(x_space, y_space, [[f(np.array([x, y])) for x in x_space] for y in y_space],
                                  levels=sorted(list(set([f(dot) for dot in dots]))))
        if clabel:
            plt.clabel(contour_set)
    if contour:
        contour_set = plt.contourf(x_space, y_space, [[f(np.array([x, y])) for x in x_space] for y in y_space],
                                   levels=30)
        plt.colorbar(contour_set)

        if clabel:
            plt.clabel(contour_set)
    if constraints:
        for constr in constraints:
            if type(constr) is LinearConstraint:
                plt.contour(x_space, y_space,
                            [[(constr.lb <= constr.A.dot(np.array([x, y]))).all()
                              and (constr.A.dot(np.array([x, y])) <= constr.ub).all() for x in x_space]
                             for y in y_space], levels=2)
            elif type(constr) is NonlinearConstraint:
                plt.contour(x_space, y_space,
                            [[(constr.lb <= constr.fun(np.array([x, y]))).all()
                              and (constr.fun(np.array([x, y])) <= constr.ub).all() for x in x_space]
                             for y in y_space], levels=2)
    if dots_show:
        x, y = dots[:, 0], dots[:, 1]
        if quiver:
            plt.quiver(x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units='xy', angles='xy', scale=1,
                       color=color, label=label, width=width)
        else:
            plt.plot(x[:], y[:], "-", color=color, label=label, alpha=alpha)
    if show:
        plt.legend(fontsize="xx-small", loc='upper right')
        plt.show()


def show_2arg_func_slice(func: Callable[[np.ndarray], float], x_min=-100, x_max=100, y_min=-100, y_max=100, **kwargs):
    grid = np.mgrid[x_min:x_max:complex(0, GRID_SIZE),
           y_min:y_max:complex(0, GRID_SIZE)].reshape(2, -1).T
    show_2arg_func(func, grid, **kwargs)


def show_ways_contour(func, ways: [np.ndarray]):
    show_2arg_func_slice(func,
                         x_min=min(map(lambda way: min(way[:, 0]), ways)),
                         x_max=max(map(lambda way: max(way[:, 0]), ways)),
                         y_min=min(map(lambda way: min(way[:, 1]), ways)),
                         y_max=max(map(lambda way: max(way[:, 1]), ways)),
                         show=False, dots_show=False, contour=True)
