import numpy as np

inputs = [1.23, 0.45, 1.51, 1.02]
weights = [[0.26, 0.88, -0.90, -0.44],
           [0.13, -0.95, 0.76, 0.22],
           [0.34, 0.66, 0.5, -0.89]]

bias = [0.3, 5.0, 2.0]

result = np.dot(inputs, weights) + bias
print(result)