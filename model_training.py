
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
x_tr = joblib.load('vectors/x_training_vector.pkl')
x_te = joblib.load('vectors/x_testing_vector.pkl')
y_tr = joblib.load('vectors/y_training_vector.pkl')
y_te = joblib.load('vectors/y_testing_vector.pkl')

print(f'x_tr shape: {x_tr.shape}')
print(f'x_te shape: {x_te.shape}')
print(f'y_tr length: {len(y_tr)}')
print(f'y_te length: {len(y_te)}')


# ## Feature Selection

print("Running feature selection...")
sel = SelectKBest(chi2, k=1000)
x_tr_s = sel.fit_transform(x_tr, y_tr)
x_te_s = sel.transform(x_te)

print(f'x_tr_s shape: {x_tr_s.shape}')
print(f'x_te_s shape: {x_te_s.shape}')

x_sample = x_tr_s[:5000]
y_sample = y_tr[:5000]

print(f'x_sample shape: {x_sample.shape}')
print(f'y_sample length: {len(y_sample)}')


# ## Grid Search Definitions

lr_grid = GridSearchCV(
    LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42),
    {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
    },
    cv = 5,
    scoring = 'f1',
    n_jobs = -1,
)

print('Defined Logistic Regression grid.')

svm_poly_grid = GridSearchCV(
    SVC(kernel='poly', probability=True, class_weight='balanced', random_state=42),
    {
        'C': [0.1, 1, 10],
        'degree': [2, 3],
        'gamma': ['scale', 'auto'],
    },
    cv=5,
    scoring='f1',
    n_jobs=-1,
)

print('Defined SVM Poly grid.')

svm_rbf_grid = GridSearchCV(
    SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
    {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
    },
    cv=5,
    scoring='f1',
    n_jobs=-1,
)

print('Defined SVM RBF grid.')

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski', 'manhattan'],
    },
    cv=5,
    scoring='f1',
    n_jobs=-1,
)

print('Defined KNN grid.')

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {
        'n_estimators': [100, 200],
        'max_depth': [None, 20],
        'min_samples_split': [2, 5],
    },
    cv=5,
    scoring='f1',
    n_jobs=-1,
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
        'estimator__alpha': [0.1, 1.0, 10.0],
    },
    cv=5,
    scoring='f1',
    n_jobs=-1,
)

print('Defined Ridge Classifier grid.')

grid_jobs = {
    "Logistic_Regression": lr_grid,
    "SVM_Poly": svm_poly_grid,
    "SVM RBF": svm_rbf_grid,
    "KNN": knn_grid,
    "Random_Forest": rf_grid,
    "Ridge": ridge_grid
}

print('Tuning models...')
for model_name, grid in tqdm(grid_jobs.items(), total=len(grid_jobs), desc='Grid search'):
    grid.fit(x_sample, y_sample)
    print(f'  Tuned: {model_name}')
print('All grid searches complete.')

print('LR best params:', lr_grid.best_params_)
print('SVM Poly best params:', svm_poly_grid.best_params_)
print('SVM RBF best params:', svm_rbf_grid.best_params_)
print('KNN best params:', knn_grid.best_params_)
print('RF best params:', rf_grid.best_params_)
print('Ridge best params:', ridge_grid.best_params_)

best_models = {
    'Logistic Regression': lr_grid.best_estimator_,
    'SVM Poly': svm_poly_grid.best_estimator_,
    'SVM RBF': svm_rbf_grid.best_estimator_,
    'KNN': knn_grid.best_estimator_,
    'Random Forest': rf_grid.best_estimator_,
    'Ridge Classifier': ridge_grid.best_estimator_,
}

for i in ['KNN', 'Random Forest', 'Ridge Classifier']:
    best_models[i] = best_models[i].set_params(n_jobs=-1)

best_models['Random Forest'] = best_models['Random Forest'].set_params(verbose=2)

print('Best model objects assembled.')

print('Training tuned models on full training set...')
for model_name, model in tqdm(best_models.items(), total=len(best_models), desc='Training base models'):
    model.fit(x_tr_s, y_tr)
    print(f'  Trained: {model_name}')
    joblib.dump(model, f'model/model_{model_name}.pkl'))

print('Building ensemble...')
ensemble = VotingClassifier(
    estimators=[
        ('lr', best_models['Logistic Regression']),
        ('svm_poly', best_models['SVM Poly']),
        ('svm_rbf', best_models['SVM RBF']),
        ('knn', best_models['KNN']),
        ('rf', best_models['Random Forest']),
        ('ridge', best_models['Ridge Classifier']),
    ],
    voting='soft',
)
ensemble.fit(x_tr_s, y_tr)
print('Ensemble training complete.')

joblib.dump(ensemble, f'models/ensemble_model.pkl')
joblib.dump(sel, f'models/sel.pkl')
print(f"Success! Model and Selector saved")


