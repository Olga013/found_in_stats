import numpy
import matplotlib.pyplot as plt

# data is of the form (ti, yi)
data = numpy.loadtxt("Data.txt", ndmin=2, converters=float, delimiter=",")


def og_model(ti, theta=2.702, x0=2):
    return numpy.sin(theta * ti) + x0


observed_data = data[:, 1]

model_predictions = [og_model(ti) for ti in data[:, 0]]

plt.scatter(observed_data, model_predictions)
plt.xlabel("Observed Data", fontsize=12)
plt.ylabel("Model Predictions", fontsize=12)
lims = [
    numpy.min([plt.xlim(), plt.ylim()]),
    numpy.max([plt.xlim(), plt.ylim()]),
]
plt.plot(lims, lims, "--", alpha=0.75, color="orange", label="y = x")
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
plt.show()
