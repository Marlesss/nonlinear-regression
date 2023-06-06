from math_func import *


def gauss_newton(dots: [(float, float)], start: np.ndarray, g_factory, log=False):
    prev = start
    while f(dots, g_factory)(prev) > EPS:
        g, g_difs = g_factory(*prev)

        if log:
            print(prev, f"mistake=={mistake(dots, g)}")

        r_apply, _ = r(g, g_difs)
        r_vector = np.array([r_apply(x, y) for (x, y) in dots])
        jacobian = np.array(jacobi(dots, g, g_difs))
        grad_f_calc = grad_f(jacobian, r_vector)
        grad_square_f_calc = grad_square_f(jacobian)

        delta = np.linalg.solve(grad_square_f_calc, -grad_f_calc)
        if all(abs(delta) < EPS):
            break
        new_args = prev + delta
        prev = new_args
    return prev
