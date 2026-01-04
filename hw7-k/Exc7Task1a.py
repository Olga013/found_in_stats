import numpy
import random
from scipy.optimize import minimize


EPS = 1e-12
random.seed(3)

def neg_log_likelihood(thetas: tuple[int, int], input: numpy.ndarray | list):
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
resampled = random.choices(input, k=len(input))

theta_0 = [0.2, 0.2]
bounds = ((0, 1), (0, 1))

resampled_mle = minimize(fun=lambda thetas: neg_log_likelihood(thetas, resampled), x0 = theta_0, bounds = bounds).x

## note that this rounds slightly differently to the provided solution... 
numpy.savetxt("Exc7Task1a.txt", resampled_mle, fmt='%1.2f')