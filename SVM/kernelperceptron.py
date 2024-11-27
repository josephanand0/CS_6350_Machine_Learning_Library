import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    X = data.iloc[:, :-1].values 
    y = data.iloc[:, -1].values  
    y = np.where(y == 0, -1, 1)  
    scaler = StandardScaler()   
    X = scaler.fit_transform(X)
    return X, y

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / gamma)

def kernel_perceptron_objective(alpha, K, y):
    return -np.sum(alpha)

def kernel_perceptron_constraint(alpha, K, y):
    return np.sum(alpha * y)

def kernel_perceptron_train(X, y, gamma, max_iter=100):
    n_samples = X.shape[0]
    alpha = np.zeros(n_samples) 

    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = gaussian_kernel(X[i], X[j], gamma)

    for iteration in range(max_iter):

        constraints = {'type': 'eq', 'fun': kernel_perceptron_constraint, 'args': (K, y)}
        bounds = [(0, None) for _ in range(n_samples)]  

        result = minimize(
            fun=kernel_perceptron_objective,
            x0=alpha,
            args=(K, y),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'disp': False}
        )

        alpha = result.x
        if result.success:
            break

    return alpha, K

def kernel_perceptron_predict(X_train, y_train, X_test, alpha, gamma):
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    K_test = np.zeros((n_test, n_train))

    for i in range(n_test):
        for j in range(n_train):
            K_test[i, j] = gaussian_kernel(X_test[i], X_train[j], gamma)

    predictions = np.sign(np.dot(K_test, alpha * y_train))
    return predictions

train_data = pd.read_csv('Datasets/train.csv', header=None)
test_data = pd.read_csv('Datasets/test.csv', header=None)
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)

gamma_values = [0.1, 0.5, 1, 5, 100]

results = {}
for gamma in gamma_values:
    alpha, K_train = kernel_perceptron_train(X_train, y_train, gamma)
    train_preds = kernel_perceptron_predict(X_train, y_train, X_train, alpha, gamma)
    test_preds = kernel_perceptron_predict(X_train, y_train, X_test, alpha, gamma)
    train_error = np.mean(train_preds != y_train)
    test_error = np.mean(test_preds != y_test)
    results[gamma] = {
        "train_error": train_error,
        "test_error": test_error
    }

for gamma, res in results.items():
    print(f"Gamma = {gamma}: Train Error = {res['train_error']}, Test Error = {res['test_error']}")
