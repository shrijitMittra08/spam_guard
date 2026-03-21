from sklearnex import patch_sklearn
patch_sklearn()

import joblib
import pandas as pd
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from tqdm import tqdm

print("Loading vectors...")
x_tr = joblib.load('../vectors/x_training_vector.pkl')
x_te = joblib.load('../vectors/x_testing_vector.pkl')
y_tr = joblib.load('../vectors/y_training_vector.pkl')
y_te = joblib.load('../vectors/y_testing_vector.pkl')

print(f'x_tr shape: {x_tr.shape}')
print(f'x_te shape: {x_te.shape}')
print(f'y_tr length: {len(y_tr)}')
print(f'y_te length: {len(y_te)}')

print("Running feature selection...")
sel = SelectKBest(chi2, k=2000)
x_tr_s = sel.fit_transform(x_tr, y_tr)
x_te_s = sel.transform(x_te)

print(f'x_tr_s shape: {x_tr_s.shape}')
print(f'x_te_s shape: {x_te_s.shape}')

x_sample = x_tr_s[:5000]
y_sample = y_tr[:5000]

print(f'x_sample shape: {x_sample.shape}')
print(f'y_sample length: {len(y_sample)}')

lr_grid = GridSearchCV(
    LogisticRegression(max_iter=4000, class_weight='balanced', random_state=42),
    {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs', 'saga'],
    },
    cv = 5,
    scoring = 'f1',
    n_jobs = -1,
    verbose=2
)

print('Defined Logistic Regression grid.')

svm_poly_grid = GridSearchCV(
    SVC(kernel='poly', probability=True, class_weight='balanced', random_state=42),
    {
        'C': [0.1, 1, 10, 100],
        'degree': [2, 3],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    },
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

print('Defined SVM Poly grid.')

svm_rbf_grid = GridSearchCV(
    SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01, 0.001],
    },
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

print('Defined SVM RBF grid.')

svm_lin_grid = GridSearchCV(
    SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42),
    {
        'C': [0.001, 0.01, 1, 10, 100],

    },
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

print('Defined SVM Linear grid.')

params1, params2 = {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'manhattan', 'euclidean'],
        'algorithm': ["auto", "ball_tree", "kd_tree", "brute"],
        'leaf_size': [20, 30, 40, 50],
        'p': [1, 2]
    }, {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'manhattan', 'euclidean'],
        'p': [1, 2]
    }

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    params2,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

print('Defined KNN grid.')

params1 = {
    'n_estimators': [100, 200, 400, 600],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced', 'balanced_subsample'],
}

params2 = {
    'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    params2,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

print('Defined Random Forest grid.')

ridge_calibrated = CalibratedClassifierCV(
    RidgeClassifier(class_weight='balanced', random_state=42),
    method='sigmoid',
    cv=5,
)

ridge_grid = GridSearchCV(
    ridge_calibrated,
    {
        'estimator__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    },
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

print('Defined Ridge Classifier grid.')

grids = {
    "Logistic Regression": lr_grid,
    "SVM Poly": svm_poly_grid,
    "SVM RBF": svm_rbf_grid,
    "SVM Linear": svm_lin_grid,
    "KNN": knn_grid,
    "Random Forest": rf_grid,
    "Ridge": ridge_grid
}

print('Tuning models on sample dataset:', flush=True)
for model_name, grid in tqdm(grids.items(), total=len(grids), desc='Tuning models'):
    print(f'  Tuning: {model_name}', flush=True)
    grid.fit(x_sample, y_sample)
    print(f'  Tuned: {model_name}')
print('Tuning models complete.')

print('LR best params:', lr_grid.best_params_)
print('SVM Poly best params:', svm_poly_grid.best_params_)
print('SVM RBF best params:', svm_rbf_grid.best_params_)
print('SVM Linear best params:', svm_lin_grid.best_params_)
print('KNN best params:', knn_grid.best_params_)
print('RF best params:', rf_grid.best_params_)
print('Ridge best params:', ridge_grid.best_params_)

cv_scores = {}

for model_name, grid in tqdm(grids.items(), total=len(grids), desc='Saving CV Scores'):
    cv_scores[model_name] = grid.best_score_

lines = []

with open('../cv_scores.txt', 'w') as f:
    for model_name, cv_score in cv_scores.items():
        lines.append(f'{model_name}: {cv_score}\n')
    f.writelines(lines)

print('Saved CV Scores')

best_models = {
    'Logistic Regression': lr_grid.best_estimator_,
    'SVM Poly': svm_poly_grid.best_estimator_,
    'SVM RBF': svm_rbf_grid.best_estimator_,
    'SVM Linear': svm_lin_grid.best_estimator_,
    'KNN': knn_grid.best_estimator_,
    'Random Forest': rf_grid.best_estimator_,
    'Ridge Classifier': ridge_grid.best_estimator_,
}

for i in ['KNN', 'Random Forest', 'Ridge Classifier']:
    best_models[i] = best_models[i].set_params(n_jobs=-1)

best_models['Random Forest'] = best_models['Random Forest'].set_params(verbose=2)

print('Best model objects assembled.')

print('Training tuned models:', flush=True)
for model_name, model in tqdm(best_models.items(), total=len(best_models), desc='Training tuned models'):
    print(f'  Training: {model_name}', flush=True)
    model.fit(x_tr_s, y_tr)
    print(f'  Trained: {model_name}')
    joblib.dump(model, f'../models/model_{model_name}.pkl')

joblib.dump(sel, f'../models/sel.pkl')
print(f"Selector saved")



