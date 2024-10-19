import pandas as pd
import numpy as np
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
    y = data['label']
    return X, y


def convert_numerical_features(X):
    for col in X.columns:
        if X[col].dtype != 'object':  
            median_val = X[col].median() 
            X[col] = np.where(X[col] > median_val, 'greater', 'less') 
    return X

def compute_entropy_weighted(y, weights):
    total_weight = np.sum(weights)
    unique_classes = np.unique(y)
    weighted_counts = np.zeros(len(unique_classes))
    
    for i, c in enumerate(unique_classes):
        class_indices = (y == c)
        weighted_counts[i] = np.sum(weights[class_indices])

    weighted_probs = weighted_counts / total_weight
    return -np.sum(weighted_probs * np.log2(weighted_probs + 1e-10))


def compute_information_gain(X, y, attr, weights):
    total_entropy = compute_entropy_weighted(y, weights)
    values, counts = np.unique(X[attr], return_counts=True)
    weighted_entropy = sum((counts[i] / len(X)) * compute_entropy_weighted(y[X[attr] == value], weights[X[attr] == value])
                           for i, value in enumerate(values))
    return total_entropy - weighted_entropy


def choose_best_attribute(X, y, weights):
    best_gain = -1
    best_attr = None
    for attr in X.columns:
        gain = compute_information_gain(X, y, attr, weights)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    return best_attr


def const_tree(X, y, weights, max_depth=1):
    if len(np.unique(y)) == 1:
        return y.mode()[0]
    
    best_attr = choose_best_attribute(X, y, weights)
    tree = {best_attr: {}}
    for value in X[best_attr].unique():
        X_subset = X[X[best_attr] == value].drop(columns=[best_attr])
        y_subset = y[X[best_attr] == value]
        subtree = y_subset.mode()[0]  
        tree[best_attr][value] = subtree
    return tree


def predict_instance(tree, instance, default_label):
    if not isinstance(tree, dict):
        return tree
    attribute = next(iter(tree))
    if attribute not in instance or instance[attribute] not in tree[attribute]:
        return default_label  
    attr_value = instance[attribute]
    return tree[attribute][attr_value]

def predict_tree(tree, X, default_label):
    return X.apply(lambda instance: predict_instance(tree, instance, default_label), axis=1)

class AdaBoost:
    def __init__(self, num_iterations, learning_rate=0.2):  
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.alphas = []
        self.stumps = []
        self.training_errors = []  
        self.test_errors = [] 
        self.stump_errors = []  
        self.default_label = 1  

    def fit(self, X, y, X_test, y_test):
        X = self.convert_to_numeric(X)
        y = y.astype(float)
        n = len(y)
        weights = np.ones(n) / n
        self.default_label = y.mode()[0]  

        for t in range(self.num_iterations):
            stump = const_tree(X, y, weights, max_depth=1)
            predictions = predict_tree(stump, X, self.default_label)
            error = np.sum(weights * (predictions != y)) / np.sum(weights) 

            if error == 0:
                alpha = 1.0
            else:
                alpha = self.learning_rate * np.log((1 - error) / max(error, 1e-10))
            self.alphas.append(alpha)

           
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  

            self.stumps.append(stump)

            
            train_predictions = self.predict(X)
            test_predictions = self.predict(X_test)
            train_error = 1 - accuracy_score(y, train_predictions)
            test_error = 1 - accuracy_score(y_test, test_predictions)
            self.training_errors.append(train_error)
            self.test_errors.append(test_error)
            self.stump_errors.append(error)

    def convert_to_numeric(self, X):
        for i in range(X.shape[1]):
            col = X.iloc[:, i]  

            if not np.issubdtype(col.dtype, np.number):
                col = pd.to_numeric(col, errors='coerce')
                col[np.isnan(col)] = 0

            X.iloc[:, i] = col

        return X

    def predict(self, X):
        n = X.shape[0]
        final_predictions = np.zeros(n)

        for alpha, stump in zip(self.alphas, self.stumps):
            predictions = predict_tree(stump, X, self.default_label) 
            predictions = predictions.apply(lambda x: 1 if x == 'yes' else -1)
            final_predictions += alpha * predictions

        return np.sign(final_predictions)


train_data, test_data = load_data('Datasets/train.csv', 'Datasets/test.csv')
X_train, y_train = get_features_and_labels(train_data)
X_test, y_test = get_features_and_labels(test_data)


X_train = convert_numerical_features(X_train)
X_test = convert_numerical_features(X_test)


y_train = y_train.apply(lambda x: 1 if x == 'yes' else -1)
y_test = y_test.apply(lambda x: 1 if x == 'yes' else -1)

num_iterations = 100
adaboost = AdaBoost(num_iterations, learning_rate=0.2)
adaboost.fit(X_train, y_train, X_test, y_test)


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_iterations + 1), adaboost.training_errors, label='Training Error', marker='o')
plt.plot(range(1, num_iterations + 1), adaboost.test_errors, label='Test Error', marker='o')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Training and Test Errors vs Iteration')
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(range(1, num_iterations + 1), adaboost.stump_errors, label='Stump Error', marker='o', color='green')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Decision Stump Errors vs Iteration')
plt.legend()

plt.tight_layout()
plt.show()
