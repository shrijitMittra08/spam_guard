
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
from sklearn.ensemble import VotingClassifier

cv_scores = {}

with open('../cv_scores.txt', 'r') as f:
    for i in f.readlines():
        cv_scores[i.split(': ')[0]] = float(i.split(': ')[1].rstrip())

for model_name, score in cv_scores.items():
    print(f'{model_name}: {score}')

cv_scores_df = pd.DataFrame({
    "Model": np.array([i for i in cv_scores.keys()]),
    "CV Score": np.array([i for i in cv_scores.values()])
})


models_df = pd.read_csv('../reports/model_report.csv')

merged_df = pd.merge(models_df, cv_scores_df, on='Model')

eval_df = merged_df[['Model', 'F1', 'CV Score', 'Loss (Log Loss)']]

eval_df['Ranking'] = eval_df['F1'] + eval_df['CV Score'] + eval_df['Loss (Log Loss)']

eval_df = eval_df.sort_values(by='Ranking', ascending=False)

print('Loading vectors:')
x_tr = joblib.load('../vectors/x_training_vector.pkl')
y_tr = joblib.load('../vectors/y_training_vector.pkl')
print('x_tr shape:', x_tr.shape)
print('y_tr shape:', y_tr.shape)

print('Running feature selection:')
sel = joblib.load('../models/sel.pkl')
x_tr_s = sel.fit_transform(x_tr, y_tr)

print('Loading models:', flush=True)

models = []

model_names = ['Random Forest', 'SVM RBF', 'SVM Poly']

for name in tqdm(model_names, total=len(model_names), desc='Loading models'):
    print(f'Loading: {name}', flush=True)
    models.append((name, joblib.load(f'../models/model_{name}.pkl')))
    print(f'Loaded: {name}')

print(models)

print('Building ensemble:')

ensemble = VotingClassifier(
    estimators=models,
    voting='soft',
    weights=[2, 1, 1]
)

ensemble.fit(x_tr_s, y_tr)

print('Ensemble training complete.')

joblib.dump(ensemble, f'../models/ensemble_model.pkl')
print('Ensemble saved')


