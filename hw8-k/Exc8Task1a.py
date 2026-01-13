import numpy

from numpy.linalg import inv

# constants
VARS = [2.27, 2.26, 1.14, 1.70, 6.5]
T = numpy.array([1, 2.5, 5, 7.5, 10])
Y = numpy.array([2.78, 6.57, 10.57, 14.9, 27.1])
THETA_0 = numpy.loadtxt("Input.txt").astype(float)
EPS = 10e-8


def xi(theta, ti) -> float:
    k1, k2 = theta
    return k1 * numpy.exp(k2 * ti)


def dxi_dk1(ti, theta) -> float:
    _, k2 = theta
    return numpy.exp(k2 * ti)


def dxi_dk2(ti, theta) -> float:
    k1, k2 = theta
    return k1 * ti * numpy.exp(k2 * ti)


def weight_matrix(vars) -> numpy.ndarray:
    w = numpy.zeros((len(vars), len(vars)))
    inv_vars = [1 / x for x in vars]
    numpy.fill_diagonal(w, inv_vars)
    return w


def jacobian(theta, t) -> numpy.ndarray:
    n_data = len(t)
    n_params = len(theta)
    jcbn = numpy.empty((n_data, n_params), dtype=float)

    for i in range(n_data):
        jcbn[(i, 0)] = float(dxi_dk1(ti=t[i], theta=theta))
        jcbn[(i, 1)] = float(dxi_dk2(ti=t[i], theta=theta))
    return jcbn


def gauss_newton_step(t, theta_s, x):
    J = jacobian(theta=theta_s, t=t)
    W = weight_matrix(
        vars=VARS
    )  # technically dont need to calculate the same matrix with every step

    return theta_s - inv(J.T @ W @ J) @ J.T @ W @ (x - Y)


def run_gauss_newton(t, theta_s):
    diff = numpy.inf
    iterations = 0
    steps = []

    while diff > EPS and iterations < 30:
        steps.append(theta_s)
        x = numpy.array([xi(theta_s, ti=ti) for ti in t])
        theta_s_plus_1 = gauss_newton_step(t=t, theta_s=theta_s, x=x)
        diff = numpy.linalg.norm(theta_s_plus_1 - theta_s, 1)
        theta_s = theta_s_plus_1
        iterations += 1

    return steps


with open("Exc8Task1a.txt", mode="w") as outfile:
    theta_s = THETA_0
    gn_steps = run_gauss_newton(t=T, theta_s=theta_s)
    theta_final = gn_steps[-1]

    outfile.write(f"{theta_final[0]:.2f}\n{theta_final[1]:.2f}\n")
