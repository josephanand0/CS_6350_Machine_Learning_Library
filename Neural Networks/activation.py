import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv("datasets/train.csv", header=None)
test_data = pd.read_csv("datasets/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

def initialize_layer(input_dim, output_dim, init_type):
    weight = torch.empty(input_dim, output_dim, dtype=torch.float32)
    bias = torch.zeros(output_dim, dtype=torch.float32)

    if init_type == "xavier":
        nn.init.xavier_uniform_(weight)
    elif init_type == "he":
        nn.init.kaiming_uniform_(weight, nonlinearity="relu")
    
    return weight, bias

def forward_layer(x, weight, bias, activation_func):
    z = torch.matmul(x, weight) + bias
    if activation_func == "tanh":
        return torch.tanh(z)
    elif activation_func == "relu":
        return torch.relu(z)
    return z

def build_network(depth, width, input_size, output_size, activation, init):
    weights = []
    biases = []
    layer_dims = [input_size] + [width] * (depth - 1) + [output_size]

    for i in range(len(layer_dims) - 1):
        w, b = initialize_layer(layer_dims[i], layer_dims[i + 1], init)
        weights.append(w)
        biases.append(b)
    
    return weights, biases

def forward_propagation(x, weights, biases, activation):
    activations = [x]
    for i in range(len(weights) - 1): 
        a = forward_layer(activations[-1], weights[i], biases[i], activation)
        activations.append(a)
    z = torch.matmul(activations[-1], weights[-1]) + biases[-1]
    activations.append(z)
    return activations

def backward_propagation(y, activations, weights, biases, activation):
    gradients = {"dW": [], "db": []}
    m = y.shape[0]
    dz = activations[-1] - y

    for i in range(len(weights) - 1, -1, -1): 
        dw = torch.matmul(activations[i].T, dz) / m
        db = torch.sum(dz, dim=0) / m
        gradients["dW"].insert(0, dw)
        gradients["db"].insert(0, db)

        if i > 0: 
            da = torch.matmul(dz, weights[i].T)
            if activation == "tanh":
                dz = da * (1 - activations[i] ** 2)
            elif activation == "relu":
                dz = da * (activations[i] > 0).float()

    return gradients

def update_weights(weights, biases, gradients, learning_rate):
    for i in range(len(weights)):
        weights[i] -= learning_rate * gradients["dW"][i]
        biases[i] -= learning_rate * gradients["db"][i]

def train_network(X_train, y_train, X_test, y_test, depth, width, activation, init, epochs=100, learning_rate=1e-3):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    weights, biases = build_network(depth, width, input_size, output_size, activation, init)

    for epoch in range(epochs):
        activations = forward_propagation(X_train, weights, biases, activation)
        gradients = backward_propagation(y_train, activations, weights, biases, activation)
        update_weights(weights, biases, gradients, learning_rate)

    train_preds = forward_propagation(X_train, weights, biases, activation)[-1].detach().numpy()
    test_preds = forward_propagation(X_test, weights, biases, activation)[-1].detach().numpy()
    train_error = mean_squared_error(y_train.numpy(), train_preds)
    test_error = mean_squared_error(y_test.numpy(), test_preds)

    return train_error, test_error

depths = [3, 5, 9]
widths = [5, 10, 25, 50, 100]
activations = {"tanh": "xavier", "relu": "he"}
epochs = 100
learning_rate = 1e-3
results = {}
for activation, init in activations.items():
    results[activation] = {}
    for depth in depths:
        for width in widths:
            train_error, test_error = train_network(
                X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
                depth, width, activation, init, epochs, learning_rate
            )
            results[activation][(depth, width)] = {"train_error": train_error, "test_error": test_error}
for activation, activation_results in results.items():
    print(f"\nResults for {activation}:")
    for (depth, width), error in activation_results.items():
        print(f"Depth: {depth}, Width: {width}, Train Error: {error['train_error']:.4f}, Test Error: {error['test_error']:.4f}")
