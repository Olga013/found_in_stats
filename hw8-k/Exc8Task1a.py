import numpy as numpy
from numpy.linalg import inv

EPS = 1e-8
MAX_ITER = 30


def xi(theta, ti) -> float:
    k1, k2 = theta
    return k1 * numpy.exp(k2 * ti)


def dxi_dk1(ti, theta) -> float:
    """Calculate value of the partial derivative w.r.t. k1 for a given theta"""
    _, k2 = theta
    return numpy.exp(k2 * ti)


def dxi_dk2(ti, theta) -> float:
    """Calculate value of the partial derivative w.r.t. k2 for a given theta"""
    k1, k2 = theta
    return k1 * ti * numpy.exp(k2 * ti)


def weight_matrix(vars) -> numpy.ndarray:
    """Returns a diagonal weight matrix from variance values."""
    inv_vars = [1 / x for x in vars]
    return numpy.diag(inv_vars)


def jacobian(theta, t) -> numpy.ndarray:
    """Computes the Jacobian matrix of xi w.r.t. theta for all t."""
    n_data = len(t)
    n_params = len(theta)
    jcbn = numpy.empty((n_data, n_params), dtype=float)

    for i in range(n_data):
        jcbn[i, 0] = dxi_dk1(ti=t[i], theta=theta)
        jcbn[i, 1] = dxi_dk2(ti=t[i], theta=theta)

    return jcbn


def gauss_newton_step(theta_s, x, y, W, t):
    """Single Gauss-Newton update step."""
    J = jacobian(theta=theta_s, t=t)
    delta = inv(J.T @ W @ J) @ J.T @ W @ (x - y)
    return theta_s - delta


def gauss_newton_alg(t, y, vars, theta_s, eps=EPS, max_iter=MAX_ITER):
    """Runs the Gauss-Newton algorithm and returns all intermediate steps."""
    W = weight_matrix(vars)
    steps = []
    diff = numpy.inf
    iterations = 0

    while diff > eps and iterations < max_iter:
        steps.append(theta_s)
        x = numpy.array([xi(theta_s, ti) for ti in t])
        theta_next = gauss_newton_step(theta_s, x, y, W, t)
        diff = numpy.linalg.norm(theta_next - theta_s, 1)
        theta_s = theta_next
        iterations += 1

    return steps


if __name__ == "__main__":
    vars = [2.27, 2.26, 1.14, 1.70, 6.5]
    t = numpy.array([1, 2.5, 5, 7.5, 10])
    y = numpy.array([2.78, 6.57, 10.57, 14.9, 27.1])
    theta_0 = numpy.loadtxt("Input.txt").astype(float)

    gn_steps = gauss_newton_alg(t=t, y=y, vars=vars, theta_s=theta_0)
    theta_final = gn_steps[-1]

    with open("Exc8Task1a.txt", mode="w") as outfile:
        outfile.write(f"{theta_final[0]:.2f}\n{theta_final[1]:.2f}\n")
