from nonlinear_regression import *
from plot import *


def test4():
    correct = [2.3, 0.23]
    start = [3.0, 0.1]
    g_factory = g_2sin
    func, _ = g_factory(*correct)
    dots = np.array([(i, func(i)) for i in np.linspace(0.5, 20, 50)])
    plt.plot(dots[:, 0], dots[:, 1], '.', color=(1, 0, 0))
    ans = powell_dog_leg(dots, np.array(start), g_factory)
    print(ans)
    print(f"Mistake of solution: {mistake(dots, g_factory(*ans)[0])}")

    x_space = np.linspace(0.5, 20, GRID_SIZE)
    show_2arg_func(None, np.array([[x, g_factory(*correct)[0](x)] for x in x_space]), show=False, color=(0, 0, 1))
    show_2arg_func(None, np.array([[x, g_factory(*ans)[0](x)] for x in x_space]), color=(0, 1, 0))


def main():
    print("nonlinear regression")
    test4()


if __name__ == "__main__":
    main()
