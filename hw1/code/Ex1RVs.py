import numpy as np 
import matplotlib.pyplot as plt

#a function to get an analytical solution from the ex. above
def get_analytical_solution(z):
    result = np.zeros(len(z))
    for i in range(len(z)):
        result[i] = (2 - 2 * z[i]) if (z[i] >= 0 and z[i] <= 1) else 0
    return result

#i.)
## IMPORT DATA FROM Input.txt
inputdata = np.loadtxt('../data/Input.txt')
#print(inputdata)
inputdata = [int(item) for item in inputdata]

## SET RANDOM SEED
np.random.seed(inputdata[0])

#Get a number of samples from an input file
n = inputdata[1]

#Sample from the uniform distribution
x = np.random.uniform(low=0.0, high=1.0, size=n)
y = np.random.uniform(low=0.0, high=1.0, size=n)
z = np.minimum(x, y)

#Get an analytical solution
z_a = np.arange(-0.5, 1.5, 0.01, dtype=float)
fz_a = get_analytical_solution(z_a)

#Read a list of the values for the random variable Z
zsamples = np.genfromtxt('../data/ZSamples.txt', delimiter=',')[: -1]

#Check if our sampled Z values are the same as Z located in the file
assert np.isclose(z, zsamples, atol=1e-02).all()

#Plot both simulated Z and the analytical solution
plt.hist(z, density=True, label='simulation')
plt.plot(z_a, fz_a, label='analytical')

plt.xlabel('z')
plt.ylabel('$f_Z(z)$')
plt.legend()
plt.show()
