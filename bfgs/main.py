from nonlinear_regression import *
from plot import *
import funcs


def test1():
    func, grad = funcs.simple()
    fufunc = lambda args: func(*args)
    way = BFGS(np.array([1, 2]), fufunc, grad)
    show_2arg_func(fufunc, way, levels=True)
    print(way)
    # correct = [2.3, 0.23]
    # start = [3.0, 0.1]
    # g_factory = g_2sin
    # func, _ = g_factory(*correct)
    # dots = np.array([(i, func(i)) for i in np.linspace(0.5, 20, 50)])
    # plt.plot(dots[:, 0], dots[:, 1], '.', color=(1, 0, 0))
    # ans = powell_dog_leg(dots, np.array(start), g_factory)
    # print(ans)
    # print(f"Mistake of solution: {mistake(dots, g_factory(*ans)[0])}")
    #
    # x_space = np.linspace(0.5, 20, GRID_SIZE)
    # show_2arg_func(None, np.array([[x, g_factory(*correct)[0](x)] for x in x_space]), show=False, color=(0, 0, 1))
    # show_2arg_func(None, np.array([[x, g_factory(*ans)[0](x)] for x in x_space]), color=(0, 1, 0))


def test2():
    correct = [2.9, 0.1]
    start = [2.9, 0.09]
    g_factory = g_2sin
    g, g_difs = g_factory(*correct)
    dots = np.array([(i, g(i)) for i in np.linspace(0.5, 20, 50)])
    min_func = f(dots, g_factory)
    # plt.plot(dots[:, 0], dots[:, 1], 'o', color=(1, 0, 0))

    way = BFGS(np.array(start), min_func, grad_f_get(dots, g_factory))
    print(way)
    solution = way[-1]
    print(f"Mistake of solution: {mistake(dots, g_factory(*solution)[0])}")

    x_space = np.linspace(0.5, 20, GRID_SIZE)
    show_2arg_func(None, np.array([[x, g_factory(*correct)[0](x)] for x in x_space]), show=False, color=(0, 0, 1))
    show_2arg_func(None, np.array([[x, g_factory(*solution)[0](x)] for x in x_space]), color=(0, 1, 0))

def main():
    print("WHO?")
    test1()
    # test2()


if __name__ == "__main__":
    main()
