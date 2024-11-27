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

def gaussian_kernel(x1, x2, gamma):
    return np.exp(-np.linalg.norm(x1 - x2)**2 / gamma)
def gaussian_dual_objective(alpha, K, y):
    return 0.5 * np.dot(alpha, np.dot(alpha * y, K)) - np.sum(alpha)

def alpha_constraints(alpha, y):
    return np.dot(alpha, y)

def run_gaussian_svm_support_vectors(X, y, C_values, gamma_values):
    results = {}
    support_vector_indices_all = {} 
    
    for gamma in gamma_values:
        results[gamma] = {}
        support_vector_indices_all[gamma] = {}
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
                options={'ftol': 1e-12, 'disp': False} 
            )
            
            alpha = result.x
            support_vector_indices = np.where(alpha > 1e-7)[0] 
            support_vector_indices_all[gamma][C] = support_vector_indices
            results[gamma][C] = len(support_vector_indices)   
    return results, support_vector_indices_all

def count_overlapping_support_vectors(support_vector_indices_all, C_fixed):
    gamma_values = list(support_vector_indices_all.keys())
    overlap_results = {}
    
    for i in range(len(gamma_values) - 1):
        gamma1 = gamma_values[i]
        gamma2 = gamma_values[i + 1]
        
        overlap_count = len(
            np.intersect1d(
                support_vector_indices_all[gamma1][C_fixed],
                support_vector_indices_all[gamma2][C_fixed]
            )
        )
        overlap_results[(gamma1, gamma2)] = overlap_count
    
    return overlap_results
train_data = pd.read_csv('train.csv', header=None)
X_train, y_train = preprocess_data(train_data)

gamma_values = [0.1, 0.5, 1, 5, 100]
C_values = [100 / 873, 500 / 873, 700 / 873]
support_vector_results, support_vector_indices_all = run_gaussian_svm_support_vectors(X_train, y_train, C_values, gamma_values)

print("Number of Support Vectors for each gamma and C:")
for gamma, gamma_results in support_vector_results.items():
    for C, num_sv in gamma_results.items():
        print(f"Gamma = {gamma}, C = {C}: Support Vectors = {num_sv}")
print("-" * 50)
C_fixed = 500 / 873
overlap_results = count_overlapping_support_vectors(support_vector_indices_all, C_fixed)

print(f"Overlapping Support Vectors for C = {C_fixed}:")
for (gamma1, gamma2), overlap_count in overlap_results.items():
    print(f"Gamma = {gamma1} and Gamma = {gamma2}: Overlapping Support Vectors = {overlap_count}")
print("-" * 50)