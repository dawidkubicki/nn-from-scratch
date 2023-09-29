import numpy as np
import math
import matplotlib.pyplot as plt



# Linear function example

# def f(x):
#     return 2*x

# x = np.arange(5)
# y = f(x)

# print(x)
# print(y)

# plt.plot(x, y)
# plt.show()

# # calculate the slope
# print(f"Slope: {(y[1]-y[0])/(x[1]-x[0])}")

# The nonlinear function example

def f(x):
    return 2*x**2

p2_delta = 0.0001

x = np.arange(0,5,0.001)
y = f(x)

for i in range(5):

    x1 = i
    x2 = x1 + p2_delta

    y1 = f(x1)
    y2 = f(x2)

    approximate_derivative = (y2-y1)/(x2-x1)
    b = y2-approximate_derivative*x2

    def tangent_line(x, approximate_derivative):
        return (approximate_derivative*x) + b

    to_plot = [x1-0.9, x1, x1+0.9]
    plt.plot(to_plot, [tangent_line(i, approximate_derivative) for i in to_plot])

    plt.plot(x, y)
plt.show()