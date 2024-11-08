import pandas as pd
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    features = data.iloc[:, :-1].to_numpy()
    labels = data.iloc[:, -1].to_numpy()
    return features, labels

train_features, train_labels = load_data('Datasets/train.csv')
test_features, test_labels = load_data('Datasets/test.csv')

def train_average_perceptron(features, labels, max_epochs, rate):
    sample_count, feature_count = features.shape
    current_weights = np.zeros(feature_count)
    cumulative_weights = np.zeros(feature_count)

    for epoch in range(max_epochs):
        shuffled_indices = np.random.permutation(sample_count)
        for i in shuffled_indices:
            prediction = np.sign(np.dot(current_weights, features[i]))
            prediction = -1 if prediction == 0 else prediction

            if prediction != labels[i]:
                current_weights += rate * labels[i] * features[i]

            cumulative_weights += current_weights

    averaged_weights = cumulative_weights / (max_epochs * sample_count)
    return averaged_weights

def classify(features, weights):
    return np.where(np.dot(features, weights) >= 0, 1, -1)

learning_rate = 0.1
epochs = 10
average_weights = train_average_perceptron(train_features, train_labels, epochs, learning_rate)
test_predictions = classify(test_features, average_weights)
test_error_rate = np.mean(test_predictions != test_labels)
print("Learned Weight Vector:", average_weights)
print("Average Prediction Error on Test Data:", test_error_rate)
