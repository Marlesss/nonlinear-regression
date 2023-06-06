import numpy as np


def m(p, f, grad_f_calced, B):
    def apply(x: np.ndarray):
        return f(p) + grad_f_calced.T.dot(x) + 1 / 2 * x.T.dot(B.dot(x))

    return apply


def rho(p, x, f, m_apply):
    return (f(x) - f(x + p)) / (m_apply(np.zeros_like(p)) - m_apply(p))

