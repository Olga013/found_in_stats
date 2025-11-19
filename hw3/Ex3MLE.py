import numpy
import matplotlib.pyplot as plt

sample = []

with open('Input.txt', 'rt', encoding='utf-8') as fp:
    while line := fp.readline(): 
         sample.append(float(line))

sample = numpy.array(sample)

# part a - plot the likelihood

n = len(sample)
x_min = numpy.min(sample)
epsilon = 1e-6
a_values = numpy.linspace(x_min - 1, 5-epsilon, 100)
likelihood = numpy.where(a_values <= x_min, (1 / (5 - a_values)) ** n, 0)

# part b - overlay data with pdf 

# The maximum likelihood estimator for a is: $\hat{a} = \min(X_i)$

def uniform_pdf(x, a, b=5):
    return numpy.where((x >= a) & (x <= b), 1 / (b - a), 0)

x_vals = numpy.linspace(x_min - 0.5, 5.5, 200)
pdf_vals = uniform_pdf(x_vals, x_min)

# part c - subsample 

numpy.random.shuffle(sample)
subsample = sample[:10]

n_subsample = len(subsample)
x_min_subsample = numpy.min(subsample)

likelihood_subsample = numpy.where(a_values <= x_min_subsample, (1 / (5 - a_values)) ** n_subsample, 0)

# set up subplots for all three parts 

fig, axs = plt.subplots(3, 1, figsize=(8, 12))

axs[0].plot(a_values, likelihood)
axs[0].set_title('Likelihood function for U(a,5)')
axs[0].set_xlabel('a')
axs[0].set_ylabel('L(a)')
axs[0].grid(True)


axs[1].hist(sample, bins=10, density=True, alpha=0.5, color='blue', label='data')
axs[1].plot(x_vals, pdf_vals, 'r-', label='PDF $\\mathcal{U}(\\hat{a}, 5)$')
axs[1].set_title('Histogram of data and PDF of $\\mathcal{U}(\\hat{a}, 5)$')
axs[1].set_xlabel('x')
axs[1].set_ylabel('Density')
axs[1].legend()
axs[1].grid(True)

axs[2].plot(a_values, likelihood_subsample)
axs[2].set_title('Likelihood function L10(a)')
axs[2].set_xlabel('a')
axs[2].set_ylabel('L10(a)')
axs[2].grid(True)

plt.show()
