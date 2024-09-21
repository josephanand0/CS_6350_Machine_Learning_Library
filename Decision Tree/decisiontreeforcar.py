import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score

# We are loading the training and test data
def load_data(train_path, test_path):
    headers = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
    train_data = pd.read_csv(train_path, names=headers)
    test_data = pd.read_csv(test_path, names=headers)
    return train_data, test_data

# Splitting the dataset into features and labels.
def get_features_and_labels(data):
    X = data.drop('label', axis=1) 
    y = data['label'] 
    return X, y

# Information Gain (IG) calculation using entropy
def compute_entropy(y):
    value_counts = y.value_counts(normalize=True)
    return -sum(p * np.log2(p) for p in value_counts if p > 0)

def compute_gini(y):
    value_counts = y.value_counts(normalize=True)
    return 1 - sum(p**2 for p in value_counts)


def compute_majority_error(y):
    majority_class_count = y.value_counts().max()
    return 1 - (majority_class_count / len(y))

# Information Gain is calculated at this step for each attribute using entropy
def compute_information_gain(X, y, attr):
    total_entropy = compute_entropy(y)  
    values, counts = np.unique(X[attr], return_counts=True)
    weighted_entropy = sum((counts[i] / len(X)) * compute_entropy(y[X[attr] == value]) for i, value in enumerate(values))
    return total_entropy - weighted_entropy

def compute_gini_gain(X, y, attr):
    total_gini = compute_gini(y)  
    values, counts = np.unique(X[attr], return_counts=True)
    weighted_gini = sum((counts[i] / len(X)) * compute_gini(y[X[attr] == value]) for i, value in enumerate(values))
    return total_gini - weighted_gini

def compute_majority_error_gain(X, y, attr):
    total_me = compute_majority_error(y) 
    values, counts = np.unique(X[attr], return_counts=True)
    weighted_me = sum((counts[i] / len(X)) * compute_majority_error(y[X[attr] == value]) for i, value in enumerate(values))
    return total_me - weighted_me

# The one that provides the highest information gain is chosen as the best attribute.
def choose_best_attribute(X, y, criterion):
    best_gain = -1
    best_attr = None
    for attr in X.columns:
        if criterion == 'information_gain':
            gain = compute_information_gain(X, y, attr)
        elif criterion == 'gini':
            gain = compute_gini_gain(X, y, attr)
        elif criterion == 'majority_error':
            gain = compute_majority_error_gain(X, y, attr)
        
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    return best_attr

# Decision tree construction
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
#It recursively traverses through the decision tree to predict the label of a particular individual instance.
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

# Training decision tree for different depths and calculating training and test errors using chosen criterion.
def evaluate_model(X_train, y_train, X_test, y_test, criterion):
    errors = []
    default_label = y_train.mode()[0]  
    for depth in range(1, 7):  
        tree = const_tree(X_train, y_train, depth=0, max_depth=depth, criterion=criterion)
        train_pred = predict_tree(tree, X_train, default_label)  
        test_pred = predict_tree(tree, X_test, default_label) 

        train_error = 1 - accuracy_score(y_train, train_pred)
        test_error = 1 - accuracy_score(y_test, test_pred)
        errors.append((train_error, test_error)) 
    return errors



train_data, test_data = load_data('datasets/car/train.csv', 'datasets/car/test.csv')
X_train, y_train = get_features_and_labels(train_data)
X_test, y_test = get_features_and_labels(test_data)

criteria = ['information_gain', 'gini', 'majority_error'] 
results = {criterion: evaluate_model(X_train, y_train, X_test, y_test, criterion) for criterion in criteria}

print(f"{'Depth':<5}{'IG Train':<15}{'IG Test':<15}{'GI Train':<15}{'GI Test':<15}{'ME Train':<15}{'ME Test':<15}")
for depth in range(1, 7):
    ig_errors = results['information_gain'][depth - 1] 
    gi_errors = results['gini'][depth - 1] 
    me_errors = results['majority_error'][depth - 1]  
    print(f"{depth:<5}"
          f"{ig_errors[0]:<15.3f}{ig_errors[1]:<15.3f}"
          f"{gi_errors[0]:<15.3f}{gi_errors[1]:<15.3f}"
          f"{me_errors[0]:<15.3f}{me_errors[1]:<15.3f}")
