#!/usr/bin/env python
# coding: utf-8
#CaleyMilanowskaNovitskaia

import numpy as np
import pandas as pd
from tqdm import tqdm
rng = np.random.default_rng(seed = 123)

def compute_delta_med(x1, x2):
    return abs(np.median(x1) - np.median(x2))

def compute_statistics(x1, x2, B, t_obs):
    n = len(x1)
    T = []
    for j in range(len(B)):
        T.append([])
        for i in tqdm(range(B[j])):
            x = x1 + x2
            rng.shuffle(x)
            x1_new = x[:n]
            x2_new = x[n:]
            T_i = compute_delta_med(x1_new, x2_new)
            T[-1].append((T_i > t_obs).item())
    return T


def compute_pi_vals(T, B):
    return [(np.array(T[i]).sum()/B[i]).item() for i in range(len(B))]


def to_output(B, pi, how='w', name='output.txt'):
    with open('output.txt', how) as f:
        for i in range(len(B)):
            line = f'{B[i]}, {pi[i]}\n'
            f.write(line)


def bootstrap_pipeline(x1, x2, B, how='w'):
    t_obs = compute_delta_med(x1, x2)
    T = compute_statistics(x1, x2, B, t_obs)
    pi = compute_pi_vals(T, B)
    to_output(B, pi, how=how, name='output.txt')

#a)

gene1_type1 = [230, -1350, -1580, -400, -760]
gene1_type2 = [970, 110, -50, -190, -200]
B = [100, 1000, 10000]

bootstrap_pipeline(gene1_type1, gene1_type2, B)

#b)

df = pd.read_excel('./DeathsByState_2019_2020.xlsx', )
n_deaths_2019 = [item.item() for item in df[df['Jahr'] == 2019]\
	.drop(columns=['Jahr', 'Bundesland', '29.02.'])\
	.sum(0).astype(int).values]
n_deaths_2020 = [item.item() for item in df[df['Jahr'] == 2020]\
	.drop(columns=['Jahr', 'Bundesland'])\
	.sum(0).astype(int).values]
B = [1000, 10000, 100000, 1000000]

bootstrap_pipeline(n_deaths_2019, n_deaths_2020, B, how='a')


print('''
For both cases a) and b) the null hypothesis of no difference
could be rejected if we take alpha=0.05, as p-values are lower
than this level of significance.
''')
