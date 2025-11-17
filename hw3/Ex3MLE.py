#!/usr/bin/env python
# coding: utf-8
#CaleyMilanowskaNovitskaia

import numpy as np
import matplotlib.pyplot as plt
inputdata = np.loadtxt('./Input.txt')


def Ln(a, b, data):
    n = len(data)
    return (1/(b - a)) ** n if a <= min(data) else 0

def ML(data):
    return min(data)

def PDF(x, a, b):
    return 1/(b - a) if (x >= a) and (x <= b) else 0

n = len(inputdata)
b = 5
a = np.arange(-5.0, 5.0, 0.01)
a_ml = ML(inputdata)

plt.figure(figsize=(12, 14))
plt.subplot(3, 1, 1)
plt.plot(a, [Ln(a_i, b, inputdata) for a_i in a], label='$L_{20}$')
plt.axvline(a_ml, color='orange', linestyle='--', label='$ML: a_{ml}$'+f'={a_ml:.2f}')
plt.ylabel(r'$L_n(a)$')
plt.xlabel('$a$')
plt.title("Likelihood")
plt.legend()


x = np.arange(0.0, 7.0, 0.01)
pdf = [PDF(x_i, a_ml, b) for x_i in x]
y_data = np.full(n, 1/(b-a_ml))

plt.subplot(3, 1, 2)
plt.hist(inputdata, density=True, alpha=0.5, label='Data hist')
plt.scatter(inputdata, y_data, color='blue', label='Data')
plt.plot(x, pdf, label='PDF')
plt.ylabel('$density$')
plt.xlabel('$x$')

plt.title("Data points and PDF")
plt.legend()

np.random.seed(0)
subsample = np.random.choice(inputdata, size=10, replace=False)
plt.subplot(3, 1, 3)
plt.plot(a, [Ln(a_i, b, subsample) for a_i in a], label='$L_{10}$')
plt.axvline(ML(subsample), color='orange', linestyle='--', label='$ML_{subsample}: a_{ml}$'+f'={ML(subsample):.2f}')
plt.ylabel(r'$L_n(a)$')
plt.xlabel('$a$')
plt.title("Likelihood subsample")
plt.legend()

plt.tight_layout(pad=3.0)
plt.show()
