import numpy as np
import pandas as pd

train_data = pd.read_csv('Datasets/train.csv')
X_train = train_data.iloc[:, :-1].values  
y_train = train_data.iloc[:, -1].values  
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
optimal_w = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
print("Optimal Weight Vector:", optimal_w)
test_data = pd.read_csv('Datasets/test.csv')
X_test = test_data.iloc[:, :-1].values  
y_test = test_data.iloc[:, -1].values  
X_test = np.c_[np.ones(X_test.shape[0]), X_test]
y_pred_test = np.dot(X_test, optimal_w)
test_cost = (1 / (2 * len(y_test))) * np.sum((y_pred_test - y_test) ** 2)

print(f"Test data cost function (analytical solution): {test_cost}")
