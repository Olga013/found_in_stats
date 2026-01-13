import numpy
import matplotlib.pyplot as plt


from Exc8Task1a import xi

THETA_STAR = numpy.loadtxt("Exc8Task1a.txt").astype(float)
VARS = numpy.array([2.27, 2.26, 1.14, 1.70, 6.5])
SD = numpy.sqrt(VARS)
T_OBS = numpy.array([1, 2.5, 5, 7.5, 10])
Y_OBS = numpy.array([2.78, 6.57, 10.57, 14.9, 27.1])
T = numpy.linspace(0, 12, 100)


y_pred = [xi(theta=THETA_STAR, ti=ti) for ti in T]

plt.plot(T, y_pred)
plt.errorbar(
    T_OBS,
    Y_OBS,
    yerr=SD,
    fmt="o",
    capsize=3,
    linestyle="none",
)
plt.show()
