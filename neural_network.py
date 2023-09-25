import numpy as np
from nnfs.datasets import spiral_data
import math


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
X, y = spiral_data(samples=100, classes=3)

dense_layer = Dense_Layer(2, 3)
dense_layer.forward(X)

activation_relu = Activation_ReLU()
activation_relu.forward(dense_layer.output)

dense_layer2 = Dense_Layer(3, 3)
dense_layer2.forward(activation_relu.output)

activation_softmax = Activation_Softmax()
activation_softmax.forward(dense_layer2.output)

loss = Loss_Categorical_Cross_Entropy()
loss = loss.calculate(activation_softmax.output, y)


# for len of class target shape = 1
sample_outputs = np.array([[0.8, 0.05, 0.15],
                  [0.3, 0.4, 0.3],
                  [0.15, 0.65, 0.1]])
class_targets = np.array([0, 0, 1])

preds = np.argmax(sample_outputs, axis=1)

acc = 0
for pred, real in zip(preds, class_targets):
    if pred == real:
        acc+=1
acc_value = (acc / len(class_targets))*100
print(f"{acc_value}% | {acc}/{len(class_targets)}")

# for len of class target shape = 1 
# sample_outputs = np.array([[0.8, 0.05, 0.15],
#                   [0.4, 0.3, 0.3],
#                   [0.15, 0.65, 0.1]])
# class_targets = np.array([[1, 0, 0],
#                           [1, 0, 0],
#                           [0, 1, 0]])

