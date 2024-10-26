from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import pandas as pd

train_cleaned = pd.read_csv('datasets/train_cleaned.csv')
test_cleaned = pd.read_csv('datasets/test_cleaned.csv')

X = train_cleaned.drop(columns=['income>50K'])
y = train_cleaned['income>50K']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

hgb_model = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=200,       
    max_depth=10,       
    random_state=42
)
hgb_model.fit(X_train, y_train)

y_val_pred_prob = hgb_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, y_val_pred_prob)
print(f"AUC on validation set: {val_auc}")
X_test = test_cleaned.drop(columns=['ID'])
test_ids = test_cleaned['ID']
test_predictions = hgb_model.predict_proba(X_test)[:, 1]
submission = pd.DataFrame({
    'ID': test_ids,
    'Prediction': test_predictions
})
submission_file_path = 'Submissions/submission_hist_gradient_boosting.csv'
submission.to_csv(submission_file_path, index=False)

