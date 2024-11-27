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

def linear_kernel(x1, x2):
    return np.dot(x1, x2.T)

def dual_objective(alpha, X, y):
    K = linear_kernel(X, X)  
    return 0.5 * np.dot(alpha, np.dot(alpha * y, K)) - np.sum(alpha)


def alpha_constraints(alpha, y):
    return np.dot(alpha, y)

def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)
def run_dual_svm(X, y, X_test, y_test, C_values):
    results = {}
    K = linear_kernel(X, X) 

    for C in C_values:
        initial_alpha = np.zeros(X.shape[0])
        bounds = [(0, C) for _ in range(len(y))]
        constraints = {'type': 'eq', 'fun': alpha_constraints, 'args': (y,)}

        result = minimize(
            fun=dual_objective,
            x0=initial_alpha,
            args=(X, y),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )

        alpha = result.x
        support_vector_indices = alpha > 1e-5
        support_vectors = X[support_vector_indices]
        support_alphas = alpha[support_vector_indices]
        support_labels = y[support_vector_indices]
        w = np.sum((support_alphas * support_labels)[:, None] * support_vectors, axis=0)
        b = np.mean(support_labels - np.dot(support_vectors, w))
        train_preds = predict(X, w, b)
        test_preds = predict(X_test, w, b)
        train_error = np.mean(train_preds != y)
        test_error = np.mean(test_preds != y_test)
        results[C] = {
            "weights": w,
            "bias": b,
            "train_error": train_error,
            "test_error": test_error,
        }
    
    return results

C_values = [100 / 873, 500 / 873, 700 / 873]

results_dual = run_dual_svm(X_train, y_train, X_test, y_test, C_values)
for C, res in results_dual.items():
    print(f"C = {C}")
    print(f"Weights (w): {res['weights']}")
    print(f"Bias (b): {res['bias']}")
    print(f"Train Error: {res['train_error']}, Test Error: {res['test_error']}")
    print("-" * 50)
