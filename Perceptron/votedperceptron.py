import pandas as pd
import numpy as np

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    features = data.iloc[:, :-1].to_numpy()
    labels = data.iloc[:, -1].to_numpy()
    return features, labels

train_features, train_labels = load_data('Datasets/train.csv')
test_features, test_labels = load_data('Datasets/test.csv')

def train_voted_perceptron(features, labels, max_epochs, rate):
    sample_count, feature_count = features.shape
    current_weights = np.zeros(feature_count)
    epoch_weights = []
    success_counts = [] 

    for epoch in range(max_epochs):
        correct_count = 0 
        shuffled_indices = np.random.permutation(sample_count)
        for i in shuffled_indices:
            prediction = np.sign(np.dot(current_weights, features[i]))
            prediction = -1 if prediction == 0 else prediction
            if prediction == labels[i]:
                correct_count += 1
            else:
                current_weights += rate * labels[i] * features[i]
        epoch_weights.append(current_weights.copy())
        success_counts.append(correct_count)

    return epoch_weights, success_counts

def predict_voted_perceptron(features, weight_list, count_list):
    predictions = []
    for x in features:
        vote_sum = sum(count * np.sign(np.dot(w, x)) for w, count in zip(weight_list, count_list))
        predictions.append(1 if vote_sum >= 0 else -1)
    return np.array(predictions)
learning_rate = 0.1
epochs = 10

voted_weights, count_per_epoch = train_voted_perceptron(train_features, train_labels, epochs, learning_rate)
for idx, (weights, count) in enumerate(zip(voted_weights, count_per_epoch), start=1):
    print(f"Weight vector {idx}: {weights}, count: {count}")

test_predictions = predict_voted_perceptron(test_features, voted_weights, count_per_epoch)
test_error_rate = np.mean(test_predictions != test_labels)
print(f"Average Test Error: {test_error_rate:.3f}")
