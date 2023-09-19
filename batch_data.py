# Batches help with generalization

import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

# Matrix Product

layer1_result = np.dot(np.array(inputs), np.array(weights).T) + biases
print(layer1_result)
print(layer1_result.shape)

output_layer_result = np.dot(layer1_result, np.array(weights2).T) + biases2
print(output_layer_result)
print(output_layer_result.shape)