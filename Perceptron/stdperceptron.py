import pandas as pd
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    features = data.iloc[:, :-1].to_numpy()
    labels = data.iloc[:, -1].to_numpy()
    return features, labels

train_features, train_labels = load_data('Datasets/train.csv')
test_features, test_labels = load_data('Datasets/test.csv')

def train_perceptron(features, labels, num_epochs, learning_rate):
    weights = np.zeros(features.shape[1])

    for epoch in range(num_epochs):
        shuffled_indices = np.random.permutation(len(features))
        shuffled_features = features[shuffled_indices]
        shuffled_labels = labels[shuffled_indices]

        for i in range(len(shuffled_features)):
            feature_vector = shuffled_features[i]
            label = shuffled_labels[i]
            if label * np.dot(weights, feature_vector) <= 0:
                weights += learning_rate * label * feature_vector

    return weights

def classify(features, weights):
    return np.where(np.dot(features, weights) >= 0, 1, -1)

num_epochs = 10
learning_rate = 0.1
weights = train_perceptron(train_features, train_labels, num_epochs, learning_rate)
test_predictions = classify(test_features, weights)
test_error_rate = np.mean(test_predictions != test_labels)
print("Learned Weight Vector:", weights)
print("Average Prediction Error on Test Data:", test_error_rate)
