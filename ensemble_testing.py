
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, precision_score, recall_score, f1_score

print('Loading testing vectors:')
x_te = joblib.load('../vectors/x_testing_vector.pkl')
y_te = joblib.load('../vectors/y_testing_vector.pkl')
print('Loaded testing vectors')

print("Loading saved selector.")
sel = joblib.load('../models/sel.pkl')
print("Loaded saved selector")

print("Applying feature selection to test data...")
x_te_s = sel.transform(x_te)

print('Loading ensemble:')
ensemble = joblib.load('../models/ensemble_model.pkl')

print('Testing ensemble')

y_pred = ensemble.predict(x_te_s)
y_prob = ensemble.predict_proba(x_te_s)

print('Accuracy:', round(accuracy_score(y_te, y_pred) * 100, 2), '%')
print('Log Loss:', round(log_loss(y_te, y_prob), 4))
print('\nClassification Report:\n', classification_report(y_te, y_pred, target_names=['Ham', 'Spam']))
print('Confusion Matrix:\n', confusion_matrix(y_te, y_pred))

with open('../reports/ensemble_report.txt', 'w') as f:
    f.write("--- Accuracy & Log Loss ---\n")
    f.write(f"Accuracy: {round(accuracy_score(y_te, y_pred) * 100, 2)}%\n")
    f.write(f"Log Loss: {round(log_loss(y_te, y_prob), 4)}\n\n")

    f.write("--- Classification Report ---\n")
    f.write(classification_report(y_te, y_pred, target_names=['Ham', 'Spam']))
    f.write("\n\n")

    f.write("--- Confusion Matrix ---\n")
    cm = confusion_matrix(y_te, y_pred)
    f.write(f"True Ham  (Correctly identified ham emails): {cm[0][0]}\n")
    f.write(f"False Spam (Incorrectly identified hams as spams): {cm[0][1]}\n")
    f.write(f"False Ham  (Incorrectly identified spams as hams): {cm[1][0]}\n")
    f.write(f"True Spam (Correctly identified spam emails): {cm[1][1]}\n")

print("Saved detailed ensemble statistics to 'ensemble_report.txt'")


