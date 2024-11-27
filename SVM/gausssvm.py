import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    X = data.iloc[:, :-1].values 
    y = data.iloc[:, -1].values  
    y = np.where(y == 0, -1, 1)  
    scaler = StandardScaler()     
    X = scaler.fit_transform(X)
    return X, y


train_data = pd.read_csv('Datasets/train.csv', header=None)
test_data = pd.read_csv('Datasets/test.csv', header=None)
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)


def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / gamma)

def gaussian_dual_objective(alpha, K, y):
    return 0.5 * np.dot(alpha, np.dot(alpha * y, K)) - np.sum(alpha)

def alpha_constraints(alpha, y):
    return np.dot(alpha, y)

def predict_gaussian(X, support_vectors, support_alphas, support_labels, b, gamma):
    K_test = np.zeros((X.shape[0], support_vectors.shape[0]))
    for i in range(X.shape[0]):
        for j in range(support_vectors.shape[0]):
            K_test[i, j] = gaussian_kernel(X[i], support_vectors[j], gamma)
    return np.sign(np.dot(K_test, support_alphas * support_labels) + b)


def run_gaussian_svm(X, y, X_test, y_test, C_values, gamma_values):
    results = {}
    
    for gamma in gamma_values:
        results[gamma] = {}
        K = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                K[i, j] = gaussian_kernel(X[i], X[j], gamma)

        for C in C_values:
            initial_alpha = np.zeros(X.shape[0]) 
            bounds = [(0, C) for _ in range(len(y))]
            constraints = {'type': 'eq', 'fun': alpha_constraints, 'args': (y,)}
            result = minimize(
                fun=gaussian_dual_objective,
                x0=initial_alpha,
                args=(K, y),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
            )
            alpha = result.x
            support_vector_indices = alpha > 1e-5
            support_vectors = X[support_vector_indices]
            support_alphas = alpha[support_vector_indices]
            support_labels = y[support_vector_indices]
            b = np.mean(support_labels - np.sum(K[support_vector_indices][:, support_vector_indices] * 
                                                (support_alphas * support_labels)[:, None], axis=0))
            train_preds = predict_gaussian(X, support_vectors, support_alphas, support_labels, b, gamma)
            test_preds = predict_gaussian(X_test, support_vectors, support_alphas, support_labels, b, gamma)
            train_error = np.mean(train_preds != y)
            test_error = np.mean(test_preds != y_test)
            results[gamma][C] = {
                "train_error": train_error,
                "test_error": test_error,
                "bias": b,
            }

    return results

gamma_values = [0.1, 0.5, 1, 5, 100]
C_values = [100 / 873, 500 / 873, 700 / 873]

results_gaussian = run_gaussian_svm(X_train, y_train, X_test, y_test, C_values, gamma_values)

for gamma, gamma_results in results_gaussian.items():
    for C, res in gamma_results.items():
        print(f"Gamma = {gamma}, C = {C}: Train Error = {res['train_error']}, Test Error = {res['test_error']}")
        print(f"Bias (b): {res['bias']}")
        print("-" * 50)
