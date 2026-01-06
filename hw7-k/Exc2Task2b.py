import numpy
import matplotlib.pyplot as plt
from Exc7Task2a import g, first_deriv_g, second_deriv_g


data = numpy.loadtxt("Data.txt", ndmin=2, converters=float, delimiter=",")

x = numpy.linspace(0, 20)
# y_g = [g(i, _,  _) for i in x]
