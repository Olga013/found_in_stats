import numpy

EPS = 10e-8

data = numpy.loadtxt("Data.txt", ndmin=2, converters=float, delimiter=",")


def g(t, y, theta, x0=2):
    x = numpy.sin(theta * t) + x0
    return ((y - x) ** 2).sum()


def first_deriv_g(t, y, theta, x0=2):
    return (-2 * t * numpy.cos(theta * t) * (y - numpy.sin(theta * t) - x0)).sum()


def second_deriv_g(t, y, theta, x0=2):
    return (
        2
        * t**2
        * (
            numpy.sin(theta * t) * (y - numpy.sin(theta * t) - x0)
            + numpy.cos(theta * t) ** 2
        )
    ).sum()


def newtons_step(theta):
    return theta - (
        first_deriv_g(t=data[:, 0], y=data[:, 1], theta=theta)
        / second_deriv_g(t=data[:, 0], y=data[:, 1], theta=theta)
    )


with open("Exc7Task2a.txt", mode="w") as outfile:
    theta_s = 3
    diff = numpy.inf
    iterations = 0

    while diff > EPS and iterations < 30:
        outfile.write("%.3f\n" % theta_s)
        theta_s_plus_1 = newtons_step(theta_s)
        diff = abs(theta_s - theta_s_plus_1)
        theta_s = theta_s_plus_1
        iterations += 1
