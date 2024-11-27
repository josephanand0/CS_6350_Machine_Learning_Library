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
