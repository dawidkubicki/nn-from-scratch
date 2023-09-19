import numpy as np
import matplotlib.pyplot as plt


sample_outputs = np.random.randn(20)
def plot_activation_result(func):
    plt.scatter([i for i in range(len(sample_outputs))], [func(output) for output in sample_outputs], s=40)
    plt.show()

# Simplest activation function -> The Step Activation Function
def step(neuron_output):
    return 1 if neuron_output >= 0 else 0

# The Sigmoid Activation Function
def sigmoid(neuron_output):
    return 1/(1+np.exp(-neuron_output))

# Try it!
plot_activation_result(sigmoid)