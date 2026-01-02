import numpy 
import matplotlib.pyplot as plt

EPS = 1e-12

def likelihood(thetas: tuple[int, int], input: numpy.ndarray):
    mapping = {
        (0, 0): thetas[0], 
        (0, 1): 1 - thetas[0],
        (1, 1): thetas[1], 
        (1, 0): 1 - thetas[1]
    }

    neg_log_lik = 0 

    for i in range(len(input)-1):
        neg_log_lik -= numpy.log(mapping[(input[i], input[i+1])] + EPS)
    
    return neg_log_lik
    
input = numpy.loadtxt("Input.txt")

theta_i = numpy.linspace(0.1, 0.9, 50)
z = numpy.zeros((50, 50))


for i in range(len(theta_i)):
    for j in range(len(theta_i)):
        neg_ll = likelihood((theta_i[i], theta_i[j]), input)
        z[i, j] = neg_ll
    
levels = numpy.arange(16, 20, 0.1)

plt.contour(theta_i, theta_i, z, levels)