import numpy as np
import matplotlib.pyplot as plt

# Helper function for plotting
sample_outputs = np.random.randn(20)
def plot_activation_result(func):
    activation_outputs = [func(output) for output in sample_outputs]
    plt.scatter([i for i in range(len(sample_outputs))], activation_outputs, c=activation_outputs, s=40, cmap='viridis')
    plt.show()

# Helper function of neuron
def neuron(input, weight, bias, activation_func):
    return activation_func(np.dot(input, weight)+bias)

# Simplest activation function -> The Step Activation Function
def step(neuron_output):
    return 1 if neuron_output >= 0 else 0

# The Sigmoid Activation Function
def sigmoid(neuron_output):
    return 1/(1+np.exp(-neuron_output))

# The Rectified Linear Activation Function
def relu(neuron_output):
    return neuron_output if neuron_output > 0 else 0

# Try it!
# plot_activation_result(relu)

print(neuron(neuron(0.25, 6.0, 0.0, relu), -1.0, 0.7, relu))
