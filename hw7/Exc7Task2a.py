#!/usr/bin/env python
# coding: utf-8


import numpy as np

def g(t, y, theta, x0=2):
    x = np.sin(theta * t) + x0
    return ((y - x)**2).sum()

def first_deriv_g(t, y, theta, x0=2):
    return (- 2 * (y - np.sin(theta * t) - x0) * np.cos(theta * t) * t).sum()

def second_deriv_g(t, y, theta, x0=2):
    return ( 2 * (y - np.sin(theta * t) - x0) * np.sin(theta * t) * (t**2) + 2 * (np.cos(theta * t)**2) * (t ** 2)).sum()

def theta_step(t, y, theta, x0=2):
    delta_theta = - first_deriv_g(t, y, theta, x0=2)/second_deriv_g(t, y, theta, x0=2)
    theta += delta_theta
    return theta


def run_newton_method(t, y, theta, x0=2, n_max=30, eps=10**(-8)):
    save(theta, how='w')
    for i in range(n_max):
        theta_new = theta_step(t, y, theta, x0=2)
        if abs(theta_new - theta) < eps:
            theta = theta_new
            break
        else:
            theta = theta_new
        save(theta, how='a')
    return theta


def save(x, how='a', name='Exc7Task2a.txt'):
    with open(name, how) as f:
        f.write('%.3f\n' % x)


n_max = 30
eps = 10**(-8)
theta = 3
data = np.loadtxt('Data.txt', ndmin=2, converters = float,delimiter=",")


_ = run_newton_method(data[:, 0], data[:, 1], theta, x0=2, n_max=n_max, eps=eps)



