import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def load_data(train_path, test_path):
    headers = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
               'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'label']
    train_data = pd.read_csv(train_path, names=headers)
    test_data = pd.read_csv(test_path, names=headers)
    return train_data, test_data

def get_features_and_labels(data):
    X = data.drop('label', axis=1)
    y = data['label'].map({'yes': 1, 'no': 0})
    return X, y

def convert_numerical_features(X):
    for col in X.columns:
        if X[col].dtype != 'object':  
            median_val = X[col].median() 
            X[col] = np.where(X[col] > median_val, 'greater', 'less')
    return X

def compute_entropy(y):
    value_counts = y.value_counts(normalize=True)
    return -sum(p * np.log2(p) for p in value_counts if p > 0)

def compute_information_gain(X, y, attr):
    total_entropy = compute_entropy(y)
    values, counts = np.unique(X[attr], return_counts=True)
    weighted_entropy = sum((counts[i] / len(X)) * compute_entropy(y[X[attr] == value]) for i, value in enumerate(values))
    return total_entropy - weighted_entropy

def choose_best_attribute(X, y):
    best_gain = -1
    best_attr = None
    for attr in X.columns:
        gain = compute_information_gain(X, y, attr)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    return best_attr

def const_tree(X, y, depth=0, max_depth=np.inf):
    if len(np.unique(y)) == 1:
        return y.iloc[0]
    if len(X.columns) == 0 or depth == max_depth:
        return y.mode()[0]
    
    best_attr = choose_best_attribute(X, y)
    tree = {best_attr: {}}
    for value in X[best_attr].unique():
        X_subset = X[X[best_attr] == value].drop(columns=[best_attr])
        y_subset = y[X[best_attr] == value]
        subtree = const_tree(X_subset, y_subset, depth + 1, max_depth)
        tree[best_attr][value] = subtree
    return tree

def predict_instance(tree, instance, default_label):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    if attribute not in instance or instance[attribute] not in tree[attribute]:
        return default_label
    attr_value = instance[attribute]
    return predict_instance(tree[attribute][attr_value], instance, default_label)

def predict_tree(tree, X, default_label):
    return X.apply(lambda instance: predict_instance(tree, instance, default_label), axis=1)

def bagging(X_train, y_train, X_test, y_test, n_trees=500):
    train_errors = []
    test_errors = []

    train_predictions = np.zeros((len(X_train), n_trees))
    test_predictions = np.zeros((len(X_test), n_trees))

    for i in range(n_trees):
        X_resampled, y_resampled = resample(X_train, y_train, replace=True)
        tree = const_tree(X_resampled, y_resampled, max_depth=np.inf)

        default_label_train = y_resampled.mode()[0]
        default_label_test = y_train.mode()[0]
        train_predictions[:, i] = predict_tree(tree, X_train, default_label_train)
        test_predictions[:, i] = predict_tree(tree, X_test, default_label_test)

        final_train_prediction = np.where(np.mean(train_predictions[:, :i+1], axis=1) > 0.5, 1, 0)
        final_test_prediction = np.where(np.mean(test_predictions[:, :i+1], axis=1) > 0.5, 1, 0)

        train_error = 1 - accuracy_score(y_train, final_train_prediction)
        test_error = 1 - accuracy_score(y_test, final_test_prediction)

        train_errors.append(train_error)
        test_errors.append(test_error)

    return train_errors, test_errors, test_predictions

def bias_variance_decomposition(X_train, y_train, X_test, y_test, n_trees=500, n_iterations=100, sample_size=1000):
    single_tree_predictions = np.zeros((len(X_test), n_iterations))
    bagged_predictions = np.zeros((len(X_test), n_iterations))
    
    for i in range(n_iterations):
     
        X_sampled, y_sampled = resample(X_train, y_train, replace=False, n_samples=sample_size)
        
        _, _, final_test_predictions = bagging(X_sampled, y_sampled, X_test, y_test, n_trees=n_trees)

      
        tree = const_tree(X_sampled, y_sampled, max_depth=np.inf)
        default_label_test = y_sampled.mode()[0]
        single_tree_predictions[:, i] = predict_tree(tree, X_test, default_label_test)
        
        bagged_predictions[:, i] = np.where(np.mean(final_test_predictions, axis=1) > 0.5, 1, 0)
        
    single_tree_bias = np.mean((np.mean(single_tree_predictions, axis=1) - y_test) ** 2)
    single_tree_variance = np.mean(np.var(single_tree_predictions, axis=1))
    single_tree_mse = single_tree_bias + single_tree_variance
 
    bagged_bias = np.mean((np.mean(bagged_predictions, axis=1) - y_test) ** 2)
    bagged_variance = np.mean(np.var(bagged_predictions, axis=1))
    bagged_mse = bagged_bias + bagged_variance
    
    
    return {
        'single_tree_bias': single_tree_bias,
        'single_tree_variance': single_tree_variance,
        'single_tree_mse': single_tree_mse,
        'bagged_bias': bagged_bias,
        'bagged_variance': bagged_variance,
        'bagged_mse': bagged_mse
    }


train_data, test_data = load_data('Datasets/train.csv', 'Datasets/test.csv')
X_train, y_train = get_features_and_labels(train_data)
X_test, y_test = get_features_and_labels(test_data)

X_train = convert_numerical_features(X_train)
X_test = convert_numerical_features(X_test)

results = bias_variance_decomposition(X_train, y_train, X_test, y_test, n_trees=500, n_iterations=100, sample_size=1000)


print("Average Single Tree Bias:", results['single_tree_bias'])
print("Average Single Tree Variance:", results['single_tree_variance'])


print("Average Bagged Trees Bias:", results['bagged_bias'])
print("Average Bagged Trees Variance:", results['bagged_variance'])

