import numpy as np
from nnfs.datasets import spiral_data, vertical_data
import math
import matplotlib.pyplot as plt


class Dense_Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01*np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values/np.sum(exp_values, axis=1, keepdims=True)

class Activation_Log_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        self.output = np.log(exp_values/np.sum(exp_values, axis=1, keepdims=True))

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss
    
class Loss_Categorical_Cross_Entropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape)==1:
            correct_pred = y_pred_clipped[[range(len(y_pred_clipped)), y_true]]
        elif len(y_true.shape)==2:
            correct_pred = np.sum(y_pred_clipped*y_true, axis=1)

        neg_log_likehood = -np.log(correct_pred)
        return neg_log_likehood

# Create dataset
# X, y = spiral_data(samples=100, classes=3)
X, y = vertical_data(samples=100, classes=3)

# plt.scatter(X[:,0], X[:, 1], c=y, cmap='brg')
# plt.show()


dense_layer = Dense_Layer(2, 3)
dense_layer2 = Dense_Layer(3, 3)
activation_relu = Activation_ReLU()
activation_softmax = Activation_Softmax()
loss_function = Loss_Categorical_Cross_Entropy()

best_dl_weights = dense_layer.weights.copy()
best_dl_biases = dense_layer.biases.copy()
best_dl_weights2 = dense_layer2.weights.copy()
best_dl_biases2 = dense_layer2.biases.copy()

learning_rate = 0.5
lowest_loss = 999999

for i in range(10000):

    dense_layer.weights += learning_rate*np.random.randn(2, 3) 
    dense_layer.biases += learning_rate*np.random.randn(1, 3) 
    dense_layer2.weights += learning_rate*np.random.randn(3, 3) 
    dense_layer2.biases += learning_rate*np.random.randn(1, 3) 

    dense_layer.forward(X)
    activation_relu.forward(dense_layer.output)
    dense_layer2.forward(activation_relu.output)
    activation_softmax.forward(dense_layer2.output)

    loss = loss_function.calculate(activation_softmax.output, y)
    preds = np.argmax(activation_softmax.output, axis=1)

    acc = 0
    if len(y.shape) == 1:
        for pred, real in zip(preds, y):
            if pred == real:
                acc+=1
        acc_value = (acc / len(y))*100

    elif len(y.shape) == 2:
        y_pos = np.argmax(y, axis=1)
        for pred, real in zip(preds, y_pos):
            if (pred==real):
                acc+=1
        acc_value = np.mean(preds==y_pos)

    if loss < lowest_loss:
        print(f"New set of weights has been found! Iteration: {i+1} Loss: {loss} Accuracy: {acc_value:.2f}% | {acc}/{len(y)}")
        best_dl_weights = dense_layer.weights.copy()
        best_dl_biases = dense_layer.biases.copy()
        best_dl_weights2 = dense_layer2.weights.copy()
        best_dl_biases2 = dense_layer2.biases.copy()
        lowest_loss = loss
    else:
        dl_weights = best_dl_weights.copy()
        dl_biases = best_dl_biases.copy()
        dl_weights2 = best_dl_weights2.copy()
        dl_biases2 = best_dl_biases2.copy()


