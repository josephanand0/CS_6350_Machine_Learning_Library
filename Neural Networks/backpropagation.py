import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    sig = sigmoid(z)
    return sig * (1 - sig)

def initialize_weights(input_size, hidden_layer1_size, hidden_layer2_size, output_size):
    np.random.seed(42) 
    W1 = np.random.randn(input_size, hidden_layer1_size) * 0.01
    b1 = np.zeros((1, hidden_layer1_size))
    W2 = np.random.randn(hidden_layer1_size, hidden_layer2_size) * 0.01
    b2 = np.zeros((1, hidden_layer2_size))
    W3 = np.random.randn(hidden_layer2_size, output_size) * 0.01
    b3 = np.zeros((1, output_size))
    return W1, b1, W2, b2, W3, b3

def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)
    return z1, a1, z2, a2, z3, a3


def backward_propagation(X, y, z1, a1, z2, a2, z3, a3, W2, W3):
    m = X.shape[0]
    output_error = a3 - y
    output_delta = output_error * sigmoid_derivative(z3)
    hidden2_error = np.dot(output_delta, W3.T)
    hidden2_delta = hidden2_error * sigmoid_derivative(z2)
    hidden1_error = np.dot(hidden2_delta, W2.T)
    hidden1_delta = hidden1_error * sigmoid_derivative(z1)
    dW3 = np.dot(a2.T, output_delta) / m
    db3 = np.sum(output_delta, axis=0, keepdims=True) / m
    dW2 = np.dot(a1.T, hidden2_delta) / m
    db2 = np.sum(hidden2_delta, axis=0, keepdims=True) / m
    dW1 = np.dot(X.T, hidden1_delta) / m
    db1 = np.sum(hidden1_delta, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3

def update_weights(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    return W1, b1, W2, b2, W3, b3

def train_neural_network(X_train, y_train, input_size, hidden_layer1_size, hidden_layer2_size, output_size, learning_rate, epochs):
    W1, b1, W2, b2, W3, b3 = initialize_weights(input_size, hidden_layer1_size, hidden_layer2_size, output_size)

    for epoch in range(epochs):
        for i in range(len(X_train)):
            X = X_train[i].reshape(1, -1)
            y = y_train[i].reshape(1, -1)
            
            z1, a1, z2, a2, z3, a3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
            
            dW1, db1, dW2, db2, dW3, db3 = backward_propagation(X, y, z1, a1, z2, a2, z3, a3, W2, W3)
            
            W1, b1, W2, b2, W3, b3 = update_weights(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)
    
    return W1, b1, W2, b2, W3, b3

def test_neural_network(X_test, y_test, W1, b1, W2, b2, W3, b3):
    correct_predictions = 0
    for i in range(len(X_test)):
        _, _, _, _, _, a3 = forward_propagation(X_test[i].reshape(1, -1), W1, b1, W2, b2, W3, b3)
        predicted_label = (a3 > 0.5).astype(int)
        if predicted_label == y_test[i]:
            correct_predictions += 1
    accuracy = correct_predictions / len(X_test)
    return accuracy
train_data = pd.read_csv("datasets/train.csv", header=None)
test_data = pd.read_csv("datasets/test.csv", header=None)
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)
input_size = X_train.shape[1]
hidden_layer1_size = 4
hidden_layer2_size = 4
output_size = 1
learning_rate = 0.01
epochs = 100

W1, b1, W2, b2, W3, b3 = train_neural_network(X_train, y_train, input_size, hidden_layer1_size, hidden_layer2_size, output_size, learning_rate, epochs)

accuracy = test_neural_network(X_test, y_test, W1, b1, W2, b2, W3, b3)
print(f"Accuracy on test data: {accuracy * 100:.2f}%")
