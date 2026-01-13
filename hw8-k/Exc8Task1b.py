import numpy
import matplotlib.pyplot as plt

from Exc8Task1a import run_gauss_newton


# values/constants
VARS = numpy.array([2.27, 2.26, 1.14, 1.70, 6.5])
T = numpy.array([1, 2.5, 5, 7.5, 10])
Y = numpy.array([2.78, 6.57, 10.57, 14.9, 27.1])
LEVELS = [1, 5, 20, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
THETA_0 = numpy.loadtxt("Input.txt").astype(float)


def neg_log_likelihood(*, yi, ti, vi, theta) -> float:
    k1, k2 = theta
    x = k1 * numpy.exp(k2 * ti)
    return (((yi - x) ** 2 / vi)).sum()


k1, k2 = numpy.meshgrid(
    numpy.linspace(0.1, 15, 200), numpy.linspace(0.05, 0.5, 200), indexing="xy"
)
z = numpy.zeros_like(k1)

for i in range(k1.shape[0]):
    for j in range(k1.shape[1]):
        z[i, j] = neg_log_likelihood(yi=Y, ti=T, vi=VARS, theta=(k1[i, j], k2[i, j]))

z_rel = z - z.min()

gn_steps = run_gauss_newton(t=T, theta_s=THETA_0)
x_steps = [step[0] for step in gn_steps]
y_steps = [step[1] for step in gn_steps]


plt.contour(k1, k2, z_rel, levels=LEVELS)
plt.xlabel("k1")
plt.ylabel("k2")
plt.plot(x_steps, y_steps, "r-")
plt.show()
