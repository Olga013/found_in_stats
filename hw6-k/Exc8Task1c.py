import numpy 

EPS = 1e-12

def likelihood(theta: float) -> float:
    """used for plotting in part d"""
    n_heads = input.sum()
    n_tails = len(input) - n_heads
    l = theta**n_heads * (1-theta)**n_tails
    return l

def log_likelihood(theta: float) -> float:
    n_heads = input.sum()
    n_tails = len(input) - n_heads
    ll = n_heads * numpy.log(theta + EPS) + n_tails * numpy.log(1 - theta + EPS)
    return ll

thetas = [0.3, 0.4, 0.5, 0.6, 0.7]
input = numpy.loadtxt("Input.txt")

log_liks = list(map(log_likelihood, thetas))

numpy.savetxt("Exc8Task1c.txt", log_liks, fmt='%1.3f')
