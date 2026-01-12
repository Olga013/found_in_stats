import numpy

from numpy.linalg import inv

# values/constants
vars = [2.27, 2.26, 1.14, 1.70, 6.5]
t = numpy.array([1, 2.5, 5, 7.5, 10]).T
y = numpy.array([2.78, 6.57, 10.57, 14.9, 27.1]).T
theta_0 = numpy.loadtxt("Input.txt", ndmin=2)


def xi(theta, ti) -> float:
    k1, k2 = theta
    return k1 * numpy.exp(k2 * ti)


def dxi_dk1(ti, theta) -> float:
    _, k2 = theta
    return numpy.exp(k2 * ti)


def dxi_dk2(ti, theta) -> float:
    k1, k2 = theta
    return k1 * ti * numpy.exp(k2 * ti)


def weight_matrix(vars):
    w = numpy.zeros((len(vars), len(vars)))
    inv_vars = [1 / x for x in vars]
    return numpy.fill_diagonal(w, inv_vars)


def jacobian(theta, t):
    n_data = len(t)
    n_params = len(theta)
    jcbn = numpy.empty((n_data, n_params))

    for i in range(n_data):
        jcbn[(i, 0)] = dxi_dk1(ti=t[i], theta=theta)
        jcbn[(i, 1)] = dxi_dk2(ti=t[i], theta=theta)
    return jcbn


def model(t, theta_s):
    J = jacobian(theta=theta_s, t=t)
    W = weight_matrix(vars=vars)

    return 


