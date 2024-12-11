import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
train_data = pd.read_csv('datasets/train_cleaned.csv')
test_data = pd.read_csv('datasets/test_cleaned.csv')
X = train_data.drop(columns=['income>50K'])
y = train_data['income>50K']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(
    kernel='rbf', 
    probability=True, 
    class_weight='balanced', 
    C=1, 
    gamma='scale', 
    random_state=42
)
svm_model.fit(X_train, y_train)
y_val_pred_prob = svm_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred_prob)
print(f"AUC on validation set (SVM): {val_auc}")
X_test = test_data.drop(columns=['ID'])
test_ids = test_data['ID']
test_predictions = svm_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    'ID': test_ids,
    'Prediction': test_predictions
})
submission_file_path = 'Submissions/submission_svm.csv'
submission.to_csv(submission_file_path, index=False)
