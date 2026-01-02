from scipy.optimize import minimize
import numpy

import Exc8Task2d

input = numpy.loadtxt("Input.txt")

theta_0 = [0.5, 0.5]
bounds = ((0, 1), (0, 1))

opt = minimize(fun=lambda thetas: Exc8Task2d.likelihood(thetas, input), x0 = theta_0, bounds = bounds)

numpy.savetxt("Exc8Task2e.txt", opt.x, fmt='%1.2f')