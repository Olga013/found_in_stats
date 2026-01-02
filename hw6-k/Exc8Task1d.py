import numpy 
import plotly.express as px

import Exc8Task1c

# log-likelihood
x = numpy.linspace(0,1)
y = numpy.array([Exc8Task1c.log_likelihood(theta=theta) for theta in x])
fig = px.line(x=x, y=y)
fig.show()


# likelihood
x = numpy.linspace(0,1)
y = numpy.array([Exc8Task1c.likelihood(theta=theta) for theta in x])
fig = px.line(x=x, y=y)
fig.show()