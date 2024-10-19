import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


train_data = pd.read_csv('Datasets/train.csv')
test_data = pd.read_csv('Datasets/test.csv')

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values


X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

def batch_gradient_descent(X, y, learning_rate, tolerance, max_iters):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    cost_history = []
    
    for iteration in range(max_iters):
        predictions = X.dot(weights)
        
        errors = predictions - y
        
        cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)
        cost_history.append(cost)
       
        gradient = (1 / n_samples) * X.T.dot(errors)
    
        new_weights = weights - learning_rate * gradient
       
        if np.linalg.norm(new_weights - weights) < tolerance:
            print(f"Converged at iteration {iteration}")
            break
     
        weights = new_weights
    
    return weights, cost_history

learning_rates = [1.0, 0.5, 0.25, 0.125]
tolerance = 1e-6
max_iterations = 100000

final_weights = None
cost_history = None
best_lr = None

for lr in learning_rates:
    print(f"Testing learning rate: {lr}")
    weights, costs = batch_gradient_descent(X_train, y_train, lr, tolerance, max_iterations)
  
    if len(costs) < max_iterations:
        final_weights = weights
        cost_history = costs
        best_lr = lr
        break

plt.plot(range(len(cost_history)), cost_history)
plt.title(f'Cost Function vs Iterations (Learning rate = {best_lr})')
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.show()

print(f"Final weights: {final_weights}")
print(f"Chosen learning rate: {best_lr}")

y_pred_test = X_test.dot(final_weights)
test_cost = (1 / (2 * len(y_test))) * np.sum((y_pred_test - y_test) ** 2)
print(f"Cost on test data: {test_cost}")
