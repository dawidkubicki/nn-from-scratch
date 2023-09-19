import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# Make them a np.arrays
inputs = np.array(inputs)
weights = np.array(weights)

bias = [2.0, 3.0, 0.5]

result = np.dot(inputs, weights.T) + bias
print(result)