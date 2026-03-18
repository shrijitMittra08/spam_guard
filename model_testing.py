#!/usr/bin/env python
# coding: utf-8

# In[8]:


import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, precision_score, recall_score, f1_score


# In[9]:


print("Loading testing vectors...")
x_te = joblib.load('vectors/x_testing_vector.pkl')
y_te = joblib.load('vectors/y_testing_vector.pkl')
print("Loaded testing vectors")


# In[10]:


print("Loading saved model and selector...")
sel = joblib.load('vectors/sel.pkl')
ensemble = joblib.load('vectors/ensemble_model.pkl')
print("Loaded saved model and selector")


# In[11]:


print("Loading individual base models...")
model_names = [
    'Logistic Regression', 'SVM Poly', 'SVM RBF', 'KNN', 'Random Forest', 'Ridge Classifier'
]

best_models = dict()

for name in model_names:
    best_models[name] = joblib.load(f'models/model_{name}.pkl')


# In[12]:


print("Applying feature selection to test data...")
x_te_s = sel.transform(x_te)


# In[13]:


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


# In[14]:


evaluation_rows = []
for model_name, model in best_models.items():
    evaluation_rows.append(evaluate_model(model_name, model, x_te_s, y_te))
evaluation_rows.append(evaluate_model('Voting Ensemble', ensemble, x_te_s, y_te))

evaluation_df = pd.DataFrame(evaluation_rows).sort_values(by='F1', ascending=False)
print('Evaluation Matrix:')
print(evaluation_df.to_string(index=False))


# In[15]:


y_pred = ensemble.predict(x_te_s)
y_prob = ensemble.predict_proba(x_te_s)

print('Accuracy:', round(accuracy_score(y_te, y_pred) * 100, 2), '%')
print('Log Loss:', round(log_loss(y_te, y_prob), 4))
print('\nClassification Report:\n', classification_report(y_te, y_pred, target_names=['Ham', 'Spam']))
print('Confusion Matrix:\n', confusion_matrix(y_te, y_pred))


# In[16]:


evaluation_df.to_csv('reports/model_report.csv', index=False)
print("Saved leaderboard to 'model_report.csv'")

with open('reports/ensemble_report.txt', 'w') as f:
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


# In[9]:


get_ipython().system('jupyter nbconvert --to script model-testing.ipynb')

