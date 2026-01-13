import numpy
import matplotlib.pyplot as plt


from Exc8Task1a import xi

theta_star = numpy.loadtxt("Exc8Task1a.txt").astype(float)
vars = numpy.array([2.27, 2.26, 1.14, 1.70, 6.5])
sd = numpy.sqrt(vars)
t_obs = numpy.array([1, 2.5, 5, 7.5, 10])
y_obs = numpy.array([2.78, 6.57, 10.57, 14.9, 27.1])
t = numpy.linspace(0, 12, 100)


y_pred = [xi(theta=theta_star, ti=ti) for ti in T]

plt.plot(t, y_pred)
plt.errorbar(
    t_obs,
    y_obs,
    yerr=sd,
    fmt="o",
    capsize=3,
    linestyle="none",
)
plt.show()
