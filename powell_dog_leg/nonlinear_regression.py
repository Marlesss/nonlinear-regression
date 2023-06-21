import numpy as np
from math_func import *
from powell_dog_leg.model import *


def powell_dog_leg(dots: [(float, float)], start: np.ndarray, g_factory, delta_start=20, delta_max=30,
                   etta=1 / 8, log=False):
    iter_count = 0
    delta = delta_start
    prev = start
    while f(dots, g_factory)(prev) > EPS:
        iter_count += 1
        g, g_difs = g_factory(*prev)

        if log:
            print(prev, f"mistake=={mistake(dots, g)}")

        r_apply, _ = r(g, g_difs)
        r_vector = np.array([r_apply(x, y) for (x, y) in dots])
        J = np.array(jacobi(dots, g, g_difs))
        grad_f_calc = grad_f(J, r_vector)
        grad_square_f_calc = grad_square_f(J)

        pb = np.linalg.solve(grad_square_f_calc, -grad_f_calc)

        if np.linalg.norm(pb) <= delta:
            p = pb
        else:
            # (|| -grad || / || J * -J.T * r || ) ** 2
            # grad = J.T * r
            # B = J.T * J
            # (r.T * J * J.T * r / (r.T * J * J.T * J * J.T * r)) * J.T * r
            pu = (np.linalg.norm(-grad_f_calc) / np.linalg.norm(J.dot(grad_f_calc))) ** 2 * -grad_f_calc
            if np.linalg.norm(pu) > delta:
                p = delta / np.linalg.norm(grad_f_calc) * -grad_f_calc
            else:
                alpha_minus_one = ((np.linalg.norm(pb - pu) ** 2 * (delta - np.linalg.norm(pu) ** 2)) ** 0.5
                                   - np.dot(pu, pb - pu)) / (np.linalg.norm(pb - pu) ** 2)
                p = pu + alpha_minus_one * (pb - pu)

        if all(abs(p) < 10 * EPS):
            break

        # delta change
        rho_calc = rho(p, prev, f(dots, g_factory), m(p, f(dots, g_factory), grad_f_calc, grad_square_f_calc))
        if rho_calc < 1 / 4:
            delta /= 4
        elif rho_calc > 3 / 4:
            delta = min(2 * delta, delta_max)

        if rho_calc > etta:
            prev = prev + p
    return prev, iter_count
