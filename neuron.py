import numpy as np

#(input) -> 1 neuron (with num of weights accordingly to input shape)

inputs = [1.0, 2.0, 3.0, 4.0]
weights = [0.2, 0.4, -0.9, 1.0]
bias = 2.0

outputs = np.dot(inputs, weights) + bias

print(outputs)