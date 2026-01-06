import numpy
import random
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from Exc7Task1a import neg_log_likelihood

random.seed(3)

input = numpy.loadtxt("Input.txt")

emp_null_theta_1 = []
emp_null_theta_2 = []
bounds = ((0, 1), (0, 1))

for i in range(1000):
    resampled = random.choices(input, k=len(input))
    theta_1_mle, theta_2_mle = minimize(
        fun=lambda thetas: neg_log_likelihood(thetas, resampled),
        x0=[0.2, 0.2],
        bounds=bounds,
    ).x
    emp_null_theta_1.append(theta_1_mle)
    emp_null_theta_2.append(theta_2_mle)

obs_theta_1_mle, obs_theta_2_mle = minimize(
    fun=lambda thetas: neg_log_likelihood(thetas, input), x0=[0.5, 0.5], bounds=bounds
).x

# null distribution for theta 1 (p_00)
plt.hist(emp_null_theta_1)
plt.title(r"Empirical null distribution for $p_{00}^*$")
plt.axvline(x=obs_theta_1_mle, color="red", label=r"$p_{00}^*$")
plt.xlabel(r"$p_{00}^*$")
plt.legend()
plt.show()

# null distribution for theta 2 (p_11)
plt.hist(emp_null_theta_2)
plt.axvline(x=obs_theta_2_mle, color="red", label=r"$p^*_{11}$")
plt.title(r"Empirical null distribution for $p^*_{11}$")
plt.xlabel(r"$p_{11}^*$")
plt.legend()
plt.show()


def ecdf(x):
    x = numpy.sort(x)
    y = numpy.arange(1, len(x) + 1) / len(x)
    return x, y


# empirical cdf for theta 1 (p_00)
x, y = ecdf(emp_null_theta_1)
plt.plot(x, y)
plt.title(r"Empirical cdf distribution for $p^*_{00}$")
plt.axvline(obs_theta_1_mle, color="red", label=r"p^*_{00}")
plt.xlabel(r"p^*_{00}")
plt.ylabel("ECDF")
plt.legend()
plt.show()

# empirical cdf for theta 2 (p_11)
x, y = ecdf(emp_null_theta_2)
plt.plot(x, y)
plt.title(r"Empirical cdf distribution for $p^*_{11}$")
plt.axvline(obs_theta_2_mle, color="red", label=r"$p^*_{11}$")
plt.xlabel(r"$p^*_{11}$")
plt.ylabel("ECDF")
plt.legend()
plt.show()

# p-value for our test of the hypothesis applied to p_00
p_val_p00 = sum(1 for b in emp_null_theta_1 if b > obs_theta_1_mle) / 1000


# p-value for our test of the hypothesis applied to p_11
p_val_p11 = sum(1 for b in emp_null_theta_2 if b > obs_theta_2_mle) / 1000


spiel = f"""For p_00, the empirical p-value of {p_val_p00} indicates that the observed transition probability is unlikely to arise from an i.i.d. sequence, providing evidence for a first-order Markov model. Assuming an alpha of 0.05 we reject our null of a simple coin flip model. 

For p_11, the empirical p-value of {p_val_p11} indicates that we do not have statistical significance, assuming an alpha of 0.05, to reject the null hypotesis that the data was generated via an simple coin flip model. 
"""

print(spiel)
