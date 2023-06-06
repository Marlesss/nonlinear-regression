import numpy as np

LOW_BOUND, HIGH_BOUND = -100, 100


def generate_function(n, cond):
    min_singular = 1
    max_singular = min_singular * cond
    s = np.diag(
        -np.sort(
            -np.insert(
                np.append(
                    (max_singular - min_singular) * np.random.sample(n - 2) + min_singular,
                    min_singular),
                0,
                max_singular)
        )
    )
    q, _ = np.linalg.qr(np.random.sample((n, n)))
    a = np.matmul(np.matmul(q, s), q.T)
    b = np.random.sample(n) * np.random.randint(LOW_BOUND, HIGH_BOUND)
    c = np.random.sample() * np.random.randint(LOW_BOUND, HIGH_BOUND)

    def grad(x: np.ndarray) -> np.ndarray:
        return np.add(2 * np.matmul(a, x.T), b)

    def f(x: np.ndarray) -> float:
        return np.dot(np.dot(x.T, a), x) + np.dot(b, x) + c

    return f, grad
