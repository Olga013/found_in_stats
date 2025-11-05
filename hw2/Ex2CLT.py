import numpy as np
import matplotlib.pyplot as plt

#i.)
## IMPORT DATA FROM Input.txt
inputdata = np.loadtxt('./data/input/Input.txt')
#print(inputdata)
inputdata = [int(item) for item in inputdata]

## SET RANDOM SEED
np.random.seed(inputdata[0])

N = np.array(inputdata[1:])
lam = 1
k = 10

#calculate mean and var
mean = np.zeros((len(N), k))
var = np.zeros((len(N)))
for j in range(len(N)):
    for i in range(k):
        mean[j, i] = np.random.poisson(lam=1.0, size=N[j]).mean()
    #we consider sample variance here where the sample consists of a set of mean(X_n)
    var[j] = mean[j].var(ddof=1)

#write into the file
with open('./data/output/MeanVar.txt', 'w') as f:
    for j in range(len(N)):
        line = ','.join(['%1.2f' % item for item in mean[j]]) + '\n'
        f.write(line)
    line = ','.join(['%1.2f' % var[i] if i < len(N) else 'nan' for i in range(k)]) + '\n'
    f.write(line)

#a)
plt.title('means')
plt.boxplot(mean.T, tick_labels=N)
plt.ylabel(r'$\bar X_n$')
plt.xlabel(r'n')
plt.show()


#As we can see here, the dispersion of means of X_n 
#decreases with the increase of n. 
#For n = 250 all points are localized in the small interval 
#in contrast to means of X_n related to n = 8, which are more spread.

#b)
#As Var(\bar X_n)) = sigma^2/n and sigma = lambda = 1: 
#Var(\bar X_n) = 1/n

plt.title('vars')
plt.plot(1/np.array(N), var, '.')
plt.plot([0, 1], [0, 1], '--')
plt.ylabel(r'$Var(\bar X_n)$')
plt.xlabel(r'$\frac{1}{n}$')
plt.show()
