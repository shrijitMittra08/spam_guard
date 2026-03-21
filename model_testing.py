
from sklearnex import patch_sklearn
patch_sklearn()

import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, precision_score, recall_score, f1_score
from tqdm import tqdm

print("Loading testing vectors...")
x_te = joblib.load('../vectors/x_testing_vector.pkl')
y_te = joblib.load('../vectors/y_testing_vector.pkl')
print("Loaded testing vectors")

print("Loading saved selector.")
sel = joblib.load('../models/sel.pkl')
print("Loaded saved selector")

print("Loading individual base models...")
model_names = [
    'Logistic Regression', 'SVM Poly', 'SVM RBF', 'SVM Linear', 'KNN', 'Random Forest', 'Ridge Classifier'
]

best_models = dict()

for name in model_names:
    print(f'Loading: {name}')
    best_models[name] = joblib.load(f'../models/model_{name}.pkl')
    print(f'Loaded: {name}')
    print()

print("Loaded individual base models.")

print("Applying feature selection to test data...")
x_te_s = sel.transform(x_te)

def evaluate_model(name, model, x_eval, y_eval):
    y_pred_local = model.predict(x_eval)
    y_prob_local = model.predict_proba(x_eval)

    return {
        'Model': name,
        'Accuracy': round(accuracy_score(y_eval, y_pred_local), 4),
        'Precision': round(precision_score(y_eval, y_pred_local), 4),
        'Recall': round(recall_score(y_eval, y_pred_local), 4),
        'F1': round(f1_score(y_eval, y_pred_local), 4),
        'Loss (Log Loss)': round(log_loss(y_eval, y_prob_local), 4),
    }

evaluation_rows = []
print('Testing models:', flush=True)
for model_name, model in tqdm(best_models.items(), total=len(best_models), desc='Testing models'):
    print(f'Testing model: {model_name}', flush=True)
    evaluation_rows.append(evaluate_model(model_name, model, x_te_s, y_te))
    print(f'Tested model: {model_name}')

evaluation_df = pd.DataFrame(evaluation_rows).sort_values(by='F1', ascending=False)

print('Evaluation Matrix:')
print(evaluation_df.to_string(index=False))

evaluation_df.to_csv('../reports/model_report.csv', index=False)
print("Saved leaderboard to 'model_report.csv'")





