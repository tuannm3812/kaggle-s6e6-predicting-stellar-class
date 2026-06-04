import json
import os

def format_source(text):
    lines = text.split('\n')
    formatted = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            formatted.append(line + '\n')
        else:
            formatted.append(line)
    if len(formatted) > 1 and formatted[-1] == "":
        formatted.pop()
        if formatted:
            formatted[-1] = formatted[-1].rstrip('\n')
    return formatted

def update_ensemble_upgrades():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks/03_model_tuning_and_ensemble.ipynb')
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    cells = data["cells"]
    
    # 1. Update Cell 1 (imports)
    cell_1_source = r"""import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import os
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from scipy.optimize import minimize

warnings.filterwarnings('ignore', category=FutureWarning)

# Target mapping variables
TARGET_MAP = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
INV_TARGET_MAP = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}"""
    cells[1]["source"] = format_source(cell_1_source)
    
    # 2. Update Cell 10 (Stacking Meta Markdown)
    cell_10_source = r"""## 4. Stacking & Threshold Tuning

### 4.1. Stacking Meta-Learner Configurations
We implement and compare multiple meta-learner configurations to stack our 6 model prediction probability matrices (18 features total) using 5-fold Stratified CV:
1. **Logistic Regression Stacker:** Baseline linear meta-learner using multinomial formulation and balanced class weights.
2. **MLP Stacker:** A regularized Multi-Layer Perceptron (MLP) Classifier to capture non-linear base model interactions.
3. **Calibrated Stackers:** We apply probability calibration (Platt Scaling and Isotonic Regression) to baseline predictions before feeding them to the Logistic Regression and MLP meta-learners.

The best configuration is selected dynamically based on out-of-fold Balanced Accuracy."""
    cells[10]["source"] = format_source(cell_10_source)
    
    # 3. Update Cell 11 (Stacking Meta Code)
    cell_11_source = r"""# Helper functions for probability calibration on a specific fold split
def calibrate_platt_split(oof_train, oof_val, y_train):
    cal_train_list, cal_val_list = [], []
    for j in range(6):
        probs_tr = oof_train[:, j*3:(j+1)*3]
        probs_va = oof_val[:, j*3:(j+1)*3]
        
        lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        lr.fit(probs_tr, y_train)
        
        cal_train_list.append(lr.predict_proba(probs_tr))
        cal_val_list.append(lr.predict_proba(probs_va))
    return np.hstack(cal_train_list), np.hstack(cal_val_list)

def calibrate_isotonic_split(oof_train, oof_val, y_train):
    cal_train_list, cal_val_list = [], []
    for j in range(6):
        probs_tr = oof_train[:, j*3:(j+1)*3]
        probs_va = oof_val[:, j*3:(j+1)*3]
        
        cal_tr = np.zeros_like(probs_tr)
        cal_va = np.zeros_like(probs_va)
        for c in range(3):
            y_binary = (y_train == c).astype(int)
            iso = IsotonicRegression(out_of_bounds='clip')
            iso.fit(probs_tr[:, c], y_binary)
            cal_tr[:, c] = iso.predict(probs_tr[:, c])
            cal_va[:, c] = iso.predict(probs_va[:, c])
            
        cal_tr = np.clip(cal_tr, 1e-15, 1 - 1e-15)
        cal_tr = cal_tr / np.sum(cal_tr, axis=1, keepdims=True)
        
        cal_va = np.clip(cal_va, 1e-15, 1 - 1e-15)
        cal_va = cal_va / np.sum(cal_va, axis=1, keepdims=True)
        
        cal_train_list.append(cal_tr)
        cal_val_list.append(cal_va)
    return np.hstack(cal_train_list), np.hstack(cal_val_list)

# Stack predictions: shape (N, 18)
Xoof = np.hstack([lgb_oof, xgb_oof, cat_oof, rf_oof, et_oof, hist_gb_oof])
Xtest = np.hstack([lgb_test, xgb_test, cat_test, rf_test, et_test, hist_gb_test])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize predictions arrays for all 6 configurations
lr_oof = np.zeros((len(y), 3))
mlp_oof = np.zeros((len(y), 3))
lr_platt_oof = np.zeros((len(y), 3))
mlp_platt_oof = np.zeros((len(y), 3))
lr_iso_oof = np.zeros((len(y), 3))
mlp_iso_oof = np.zeros((len(y), 3))

lr_meta = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=42)
mlp_meta = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01, early_stopping=True, validation_fraction=0.1, random_state=42, max_iter=500)

print("Evaluating Stacker Configurations via Nested 5-Fold Stratified CV...")
for fold, (train_idx, val_idx) in enumerate(skf.split(Xoof, y), 1):
    X_tr, y_tr = Xoof[train_idx], y[train_idx]
    X_va, y_va = Xoof[val_idx], y[val_idx]
    
    # 1. Raw stackers
    lr_meta.fit(X_tr, y_tr)
    lr_oof[val_idx] = lr_meta.predict_proba(X_va)
    
    mlp_meta.fit(X_tr, y_tr)
    mlp_oof[val_idx] = mlp_meta.predict_proba(X_va)
    
    # 2. Platt calibrated stackers
    X_tr_platt, X_va_platt = calibrate_platt_split(X_tr, X_va, y_tr)
    
    lr_meta.fit(X_tr_platt, y_tr)
    lr_platt_oof[val_idx] = lr_meta.predict_proba(X_va_platt)
    
    mlp_meta.fit(X_tr_platt, y_tr)
    mlp_platt_oof[val_idx] = mlp_meta.predict_proba(X_va_platt)
    
    # 3. Isotonic calibrated stackers
    X_tr_iso, X_va_iso = calibrate_isotonic_split(X_tr, X_va, y_tr)
    
    lr_meta.fit(X_tr_iso, y_tr)
    lr_iso_oof[val_idx] = lr_meta.predict_proba(X_va_iso)
    
    mlp_meta.fit(X_tr_iso, y_tr)
    mlp_iso_oof[val_idx] = mlp_meta.predict_proba(X_va_iso)

# Compute validation scores
scores = {
    'lr': (balanced_accuracy_score(y, np.argmax(lr_oof, axis=1)), lr_oof),
    'mlp': (balanced_accuracy_score(y, np.argmax(mlp_oof, axis=1)), mlp_oof),
    'lr_platt': (balanced_accuracy_score(y, np.argmax(lr_platt_oof, axis=1)), lr_platt_oof),
    'mlp_platt': (balanced_accuracy_score(y, np.argmax(mlp_platt_oof, axis=1)), mlp_platt_oof),
    'lr_iso': (balanced_accuracy_score(y, np.argmax(lr_iso_oof, axis=1)), lr_iso_oof),
    'mlp_iso': (balanced_accuracy_score(y, np.argmax(mlp_iso_oof, axis=1)), mlp_iso_oof)
}

print("\n--- Out-of-Fold Balanced Accuracy Scores ---")
for name, (score, _) in scores.items():
    print(f"{name.upper():<12}: {score:.5f}")

best_name = max(scores, key=lambda k: scores[k][0])
best_score, best_oof = scores[best_name]
print(f"\n=== Dynamic Selection: Best Meta-Learner configuration is '{best_name}' ===")
print(f"Best OOF Balanced Accuracy Score: {best_score:.5f}")

# --- Generate Final Test Predictions using full dataset ---
if 'platt' in best_name:
    best_train, best_test = calibrate_platt_split(Xoof, Xtest, y)
    best_model = mlp_meta if 'mlp' in best_name else lr_meta
    best_model.fit(best_train, y)
    stacked_test_prob = best_model.predict_proba(best_test)
elif 'iso' in best_name:
    best_train, best_test = calibrate_isotonic_split(Xoof, Xtest, y)
    best_model = mlp_meta if 'mlp' in best_name else lr_meta
    best_model.fit(best_train, y)
    stacked_test_prob = best_model.predict_proba(best_test)
else:
    best_model = mlp_meta if 'mlp' in best_name else lr_meta
    best_model.fit(Xoof, y)
    stacked_test_prob = best_model.predict_proba(Xtest)

meta_oof = best_oof
stacked_score = best_score"""
    cells[11]["source"] = format_source(cell_11_source)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(f"Successfully updated {path}!")

if __name__ == "__main__":
    update_ensemble_upgrades()
