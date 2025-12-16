import numpy as np
from scipy.optimize import minimize

def log_likelihood(seq, theta):
    mapping = {(0, 0): theta[0],
               (0, 1): 1 - theta[0],
               (1, 0): 1 - theta[1],
               (1, 1): theta[1]}

    l = 0
    eps = 1e-12
    for i in range(len(seq) - 1):
        transition = tuple(tuple(x.item() for x in seq[i:i+2]))
        l += -np.log(mapping[transition] + eps)
        
    return l

sequence = []

with open('Input.txt', 'rt', encoding='utf-8') as fp:
    while line := fp.readline(): 
         sequence.append(float(line))

sequence = np.array(sequence)

theta0 = [0.5, 0.5]
bounds = ((0, 1), (0, 1))

res = minimize(lambda theta: log_likelihood(sequence, theta), theta0, bounds=bounds, method='SLSQP')

np.savetxt('Exc8Task2e.txt', res.x, delimiter=',', fmt='%1.2f')
