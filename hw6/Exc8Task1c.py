import numpy as np

def log_likelihood(seq, theta):
    l = 0
    for term in seq:
        l += term * np.log(theta) if term == 1 else (1 - term) * np.log(1 - theta)
    return l

sequence = []

with open('Input.txt', 'rt', encoding='utf-8') as fp:
    while line := fp.readline(): 
         sequence.append(float(line))

sequence = np.array(sequence)

thetas = [0.3, 0.4, 0.5, 0.6, 0.7]

ls = np.array(list(map(lambda theta: log_likelihood(sequence, theta = theta), thetas)))

np.savetxt('Exc8Task1c.txt', ls, delimiter=',', fmt='%1.3f')
