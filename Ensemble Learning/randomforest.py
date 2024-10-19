import pandas as pd
import numpy as np
from collections import Counter
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
    y = data['label'].apply(lambda x: 1 if x == 'yes' else 0) 
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

def compute_gini(y):
    value_counts = y.value_counts(normalize=True)
    return 1 - sum(p**2 for p in value_counts)

def compute_majority_error(y):
    majority_class_count = y.value_counts().max()
    return 1 - (majority_class_count / len(y))


def compute_information_gain(X, y, attr):
    total_entropy = compute_entropy(y)
    values, counts = np.unique(X[attr], return_counts=True)
    weighted_entropy = sum((counts[i] / len(X)) * compute_entropy(y[X[attr] == value]) for i, value in enumerate(values))
    return total_entropy - weighted_entropy

def choose_best_attribute(X, y, criterion):
    best_gain = -1
    best_attr = None
    for attr in X.columns:
        if criterion == 'information_gain':
            gain = compute_information_gain(X, y, attr)
        elif criterion == 'gini':
            gain = compute_gini(X, y, attr)
        elif criterion == 'majority_error':
            gain = compute_majority_error(X, y, attr)
        
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    return best_attr


def const_tree(X, y, depth=0, max_depth=np.inf, criterion='information_gain'):
    if len(np.unique(y)) == 1:
        return y.iloc[0]
    if len(X.columns) == 0 or depth == max_depth:
        return y.mode()[0]
    
    best_attr = choose_best_attribute(X, y, criterion)
    tree = {best_attr: {}}
    for value in X[best_attr].unique():
        X_subset = X[X[best_attr] == value].drop(columns=[best_attr])
        y_subset = y[X[best_attr] == value]
        subtree = const_tree(X_subset, y_subset, depth + 1, max_depth, criterion)
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

def evaluate_model(X_train, y_train, X_test, y_test, criterion):
    errors = []
    default_label = y_train.mode()[0]
    for depth in range(1, 17): 
        tree = const_tree(X_train, y_train, depth=0, max_depth=depth, criterion=criterion)
        train_pred = predict_tree(tree, X_train, default_label)
        test_pred = predict_tree(tree, X_test, default_label)

        train_error = 1 - accuracy_score(y_train, train_pred)
        test_error = 1 - accuracy_score(y_test, test_pred)
        errors.append((train_error, test_error))
    return errors

class RandomForestClassifier:
    def __init__(self, num_trees, max_features, max_depth=np.inf):
        self.num_trees = num_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.num_trees):
        
            selected_features = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X.iloc[:, selected_features]

            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_bootstrap = X_subset.iloc[indices]
            y_bootstrap = y.iloc[indices]

           
            tree = const_tree(X_bootstrap, y_bootstrap, max_depth=self.max_depth)
            self.trees.append((tree, selected_features))

    def predict(self, X):
    
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, (tree, selected_features) in enumerate(self.trees):
            X_subset = X.iloc[:, selected_features]
            predictions[:, i] = predict_tree(tree, X_subset, y_train.mode()[0])

        return np.where(np.mean(predictions, axis=1) > 0.5, 1, 0)

train_data, test_data = load_data('Datasets/train.csv', 'Datasets/test.csv')
X_train, y_train = get_features_and_labels(train_data)
X_test, y_test = get_features_and_labels(test_data)

X_train = convert_numerical_features(X_train)
X_test = convert_numerical_features(X_test)

num_trees_range = range(1, 11)
max_features_range = [2, 4, 6]

train_errors_rf = {2: [], 4: [], 6: []}
test_errors_rf = {2: [], 4: [], 6: []}

for max_features in max_features_range:
    for num_trees in num_trees_range:
        rf_classifier = RandomForestClassifier(num_trees=num_trees, max_features=max_features, max_depth=10)
        rf_classifier.fit(X_train, y_train)

        y_train_pred = rf_classifier.predict(X_train)
        y_test_pred = rf_classifier.predict(X_test)

        train_error = 1 - accuracy_score(y_train, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)

        train_errors_rf[max_features].append(train_error)
        test_errors_rf[max_features].append(test_error)

plt.figure(figsize=(12, 6))
for max_features in max_features_range:
    plt.plot(num_trees_range, train_errors_rf[max_features], label=f'Train Error (max_features={max_features})')
    plt.plot(num_trees_range, test_errors_rf[max_features], label=f'Test Error (max_features={max_features})')

plt.xlabel('Number of Trees')
plt.ylabel('Error')
plt.title('Training and Test Errors vs. Number of Trees (Random Forest)')
plt.legend()
plt.show()
