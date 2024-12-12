import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

train_data = pd.read_csv("datasets/train.csv", header=None)
test_data = pd.read_csv("datasets/test.csv", header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss_and_gradient_ml(X, y, weights):
    m = X.shape[0]
    predictions = sigmoid(np.dot(X, weights))
    loss = -np.mean(y * np.log(predictions + 1e-10) + (1 - y) * np.log(1 - predictions + 1e-10))
    gradient = np.dot(X.T, (predictions - y)) / m
    return loss, gradient

def train_logistic_regression_ml(X_train, y_train, X_test, y_test, gamma_0, d, epochs):
    input_size = X_train.shape[1]
    weights = np.zeros(input_size)
    losses = []

    for epoch in range(epochs):
        permutation = np.random.permutation(len(X_train))
        X_train = X_train[permutation]
        y_train = y_train[permutation]

        for t, (X, y) in enumerate(zip(X_train, y_train)):
            gamma_t = gamma_0 / (1 + (gamma_0 / d) * t)

            X = X.reshape(1, -1)
            y = np.array([y])
            loss, gradient = compute_loss_and_gradient_ml(X, y, weights)

            weights -= gamma_t * gradient

            if t % 10 == 0:
                losses.append(loss)
    train_predictions = sigmoid(np.dot(X_train, weights)) >= 0.5
    test_predictions = sigmoid(np.dot(X_test, weights)) >= 0.5
    train_error = 1 - accuracy_score(y_train, train_predictions)
    test_error = 1 - accuracy_score(y_test, test_predictions)

    return {"train_error": train_error, "test_error": test_error, "losses": losses}

gamma_0 = 0.1
d = 0.01
epochs = 100
variances = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
results_ml = {}
for variance in variances:
    results = train_logistic_regression_ml(X_train, y_train, X_test, y_test, gamma_0, d, epochs)
    results_ml[variance] = results


for v, res in results_ml.items():
    print(f"Variance: {v}, Train Error: {res['train_error']:.4f}, Test Error: {res['test_error']:.4f}")
