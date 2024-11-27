import pandas as pd
import numpy as np

def preprocess_data(data):
    X = data.iloc[:, :-1].values  
    y = data.iloc[:, -1].values  
    y = np.where(y == 0, -1, 1) 
    return X, y

train_data = pd.read_csv('Datasets/train.csv', header=None)
test_data = pd.read_csv('Datasets/test.csv', header=None)
X_train, y_train = preprocess_data(train_data)
X_test, y_test = preprocess_data(test_data)

class SVM:
    def __init__(self, C, gamma0, a):
        self.C = C
        self.gamma0 = gamma0
        self.a = a
        self.weights = None
        self.bias = 0

    def fit(self, X, y, epochs=100):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        updates = 0
        model_parameters = []

        for t in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            for i, x_i in enumerate(X):
                gamma_t = self.gamma0 / (1 + (self.gamma0 / self.a) * updates)
                if y[i] * (np.dot(self.weights, x_i) + self.bias) < 1:
                    self.weights = (1 - gamma_t) * self.weights + gamma_t * self.C * y[i] * x_i
                    self.bias += gamma_t * self.C * y[i]
                else:
                    self.weights *= (1 - gamma_t)
                updates += 1
                model_parameters.append({
                    "epoch": t,
                    "update": updates,
                    "weights": self.weights.copy(),
                    "bias": self.bias,
                    "learning_rate": gamma_t
                })

        return model_parameters

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)

C_values = [100/873, 500/873, 700/873]
gamma0 = 0.1
a = 0.01
epochs = 100

results_schedule1 = {}

for C in C_values:
    svm = SVM(C=C, gamma0=gamma0, a=a)
    model_parameters = svm.fit(X_train, y_train, epochs=epochs)
    train_error = np.mean(svm.predict(X_train) != y_train)
    test_error = np.mean(svm.predict(X_test) != y_test)
    results_schedule1[C] = {
        "train_error": train_error,
        "test_error": test_error,
        "weights": svm.weights,
        "bias": svm.bias,
        "model_parameters": model_parameters
    }

for C, res in results_schedule1.items():
    print(f"C={C}: Train Error={res['train_error']}, Test Error={res['test_error']}")
    print(f"Weights: {res['weights']}, Final Bias: {res['bias']}")


class SVM:
    def __init__(self, C, gamma0):
        self.C = C
        self.gamma0 = gamma0
        self.weights = None
        self.bias = 0

    def fit(self, X, y, epochs=100):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        updates = 0
        model_parameters = []

        for t in range(epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            for i, x_i in enumerate(X):
                gamma_t = self.gamma0 / (1 + updates)
                if y[i] * (np.dot(self.weights, x_i) + self.bias) < 1:
                    self.weights = (1 - gamma_t) * self.weights + gamma_t * self.C * y[i] * x_i
                    self.bias += gamma_t * self.C * y[i]
                else:
                    self.weights *= (1 - gamma_t)
                updates += 1
                model_parameters.append({
                    "epoch": t,
                    "update": updates,
                    "weights": self.weights.copy(),
                    "bias": self.bias,
                    "learning_rate": gamma_t
                })

        return model_parameters

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
C_values = [100/873, 500/873, 700/873]
gamma0 = 0.1
epochs = 100

results_schedule2 = {}

for C in C_values:
    svm = SVM(C=C, gamma0=gamma0)
    model_parameters = svm.fit(X_train, y_train, epochs=epochs)
    train_error = np.mean(svm.predict(X_train) != y_train)
    test_error = np.mean(svm.predict(X_test) != y_test)
    results_schedule2[C] = {
        "train_error": train_error,
        "test_error": test_error,
        "weights": svm.weights,
        "bias": svm.bias,
        "model_parameters": model_parameters
    }

for C, res in results_schedule2.items():
    print(f"C={C}: Train Error={res['train_error']}, Test Error={res['test_error']}")
    print(f"Weights: {res['weights']}, Final Bias: {res['bias']}")

comparison = {}

for C in C_values:
    weights_diff = np.linalg.norm(results_schedule1[C]["weights"] - results_schedule2[C]["weights"])
    bias_diff = abs(results_schedule1[C]["bias"] - results_schedule2[C]["bias"])
    train_error_diff = results_schedule1[C]["train_error"] - results_schedule2[C]["train_error"]
    test_error_diff = results_schedule1[C]["test_error"] - results_schedule2[C]["test_error"]
    comparison[C] = {
        "weight_difference": weights_diff,
        "bias_difference": bias_diff,
        "train_error_difference": train_error_diff,
        "test_error_difference": test_error_diff
    }

print("Comparison of Results Between Schedule 1 and Schedule 2:")
for C, comp in comparison.items():
    print(f"C = {C}:")
    print(f"  Weight Difference: {comp['weight_difference']}")
    print(f"  Bias Difference: {comp['bias_difference']}")
    print(f"  Train Error Difference: {comp['train_error_difference']}")
    print(f"  Test Error Difference: {comp['test_error_difference']}")
    print("-" * 50)
