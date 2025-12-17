import numpy as np
import matplotlib.pyplot as plt

sequence = []
with open('Input.txt', 'rt', encoding='utf-8') as fp:
    for line in fp:
        sequence.append(int(line.strip()))
sequence = np.array(sequence)

def neg_log_likelihood(seq, theta1, theta2):
    mapping = {
        (0, 0): 1 - theta1,
        (0, 1): theta1,
        (1, 0): 1 - theta2,
        (1, 1): theta2
    }
    eps = 1e-12  
    nll = 0
    for i in range(len(seq) - 1):
        transition = (seq[i], seq[i+1])
        nll -= np.log(mapping[transition] + eps)
    return nll

theta_vals = np.linspace(0.1, 0.9, 50)
Z = np.zeros((len(theta_vals), len(theta_vals)))

for i, t1 in enumerate(theta_vals):
    for j, t2 in enumerate(theta_vals):
        Z[j, i] = neg_log_likelihood(sequence, t1, t2)

X, Y = np.meshgrid(theta_vals, theta_vals)
levels = np.arange(16, 20, 0.1)
plt.contour(X, Y, Z, levels=levels)
plt.xlabel(r'$\theta_1 = p_{01}$')
plt.ylabel(r'$\theta_2 = p_{11}$')
plt.title('Negative Log-Likelihood Contour')
plt.show()