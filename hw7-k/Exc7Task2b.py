import numpy
import matplotlib.pyplot as plt
from Exc7Task2a import g, first_deriv_g, second_deriv_g


data = numpy.loadtxt("Data.txt", ndmin=2, converters=float, delimiter=",")
# data is of the form (ti, yi)

theta_vals = numpy.linspace(0, 20, 100)
t = data[:, 0]
y = data[:, 1]

g_vals = [g(t=t, y=y, theta=i) for i in theta_vals]
g_prime_vals = numpy.array([first_deriv_g(t=t, y=y, theta=i) for i in theta_vals])
g_double_prime_vals = numpy.array(
    [second_deriv_g(t=t, y=y, theta=i) for i in theta_vals]
)

plt.plot(theta_vals, g_vals, label="g")
plt.plot(theta_vals, g_prime_vals, label="g'")
plt.plot(theta_vals, g_double_prime_vals, label="g''")
plt.axhline(y=0)
plt.legend()
plt.show()

# conditions for optimal value of g is that g' is 0 and g'' is positive

# find the positions where the first derivative intersects the x axis
# i think the easiest way to do this is to find where the sign flips between successive value of g'
# i take the first index out of the sign flip arbitrarily
zero_crossings_idx = numpy.where(
    numpy.sign(g_prime_vals[:-1]) != numpy.sign(g_prime_vals[1:])
)[0]

# find the idx for which g'' is also positive
minima_idx = zero_crossings_idx[g_double_prime_vals[zero_crossings_idx] > 0]

# theta values corresponding to all optima
print(
    "The theta values corresponding to all optima are:",
    numpy.array2string(theta_vals[minima_idx], precision=3, separator=", "),
)
