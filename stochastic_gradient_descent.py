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


def sgd(X, y, learning_rate, max_iters, tolerance):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    cost_history = []
    
    for iteration in range(max_iters):
        
        shuffled_indices = np.random.permutation(n_samples)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        
        for i in range(n_samples):
            xi = X_shuffled[i, :]
            yi = y_shuffled[i]

            prediction = np.dot(xi, weights)
    
            error = prediction - yi
   
            gradient = xi * error

            weights = weights - learning_rate * gradient
            
   
            predictions_all = X.dot(weights)
            cost = (1 / (2 * n_samples)) * np.sum((predictions_all - y) ** 2)
            cost_history.append(cost)
        
        if iteration > 0 and abs(cost_history[-2] - cost_history[-1]) < tolerance:
            print(f"Convergence achieved at iteration {iteration}")
            break
    
    return weights, cost_history


learning_rate = 0.01
max_iterations = 10000
tolerance = 1e-6

final_weights, cost_history = sgd(X_train, y_train, learning_rate, max_iterations, tolerance)


plt.plot(range(len(cost_history)), cost_history)
plt.xlabel('Number of Updates')
plt.ylabel('Cost Function')
plt.title('Cost Function vs. Updates (SGD)')
plt.show()


y_pred_test = np.dot(X_test, final_weights)
test_cost = (1 / (2 * len(y_test))) * np.sum((y_pred_test - y_test) ** 2)

print(f"Final weight vector: {final_weights}")
print(f"Learning rate: {learning_rate}")
print(f"Test data cost function: {test_cost}")
