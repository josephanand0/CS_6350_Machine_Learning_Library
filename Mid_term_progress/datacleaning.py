import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_data(train_file_path, test_file_path, scale_numerical=True):
    train_data = pd.read_csv(train_file_path)
    test_data = pd.read_csv(test_file_path)
    X = train_data.drop(columns=['income>50K'])
    y = train_data['income>50K']
    categorical_columns = X.select_dtypes(include=['object']).columns
    numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns
    for col in categorical_columns:
        freq_encoding = X[col].value_counts() / len(X)
        X[col] = X[col].map(freq_encoding)
        test_data[col] = test_data[col].map(freq_encoding).fillna(0) 
    if scale_numerical:
        scaler = StandardScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
        test_data[numerical_columns] = scaler.transform(test_data[numerical_columns])
    train_cleaned = X.copy()
    train_cleaned['income>50K'] = y.reset_index(drop=True)
    train_cleaned.to_csv('datasets/train_cleaned.csv', index=False)
    test_data.to_csv('datasets/test_cleaned.csv', index=False)

clean_data('datasets/train_final.csv', 'datasets/test_final.csv', scale_numerical=True)
