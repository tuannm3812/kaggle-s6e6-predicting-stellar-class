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

def find_cell_by_id(cells, cell_id):
    for idx, cell in enumerate(cells):
        if cell.get("id") == cell_id:
            return idx
    raise ValueError(f"Cell with id '{cell_id}' not found!")

def update_blender():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks/03_model_tuning_and_ensemble.ipynb')
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    cells = data["cells"]
    
    # 1. Update imports_code
    imports_idx = find_cell_by_id(cells, "imports_code")
    imports_source = r"""import numpy as np
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
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore', category=FutureWarning)

# Target mapping variables
TARGET_MAP = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
INV_TARGET_MAP = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}"""
    cells[imports_idx]["source"] = format_source(imports_source)
    
    # 2. Update stacking_meta_md
    stacking_md_idx = find_cell_by_id(cells, "stacking_meta_md")
    stacking_md_source = r"""## 4. Stacking & Threshold Tuning

### 4.1. Stacking Meta-Learner Configurations
We implement and compare multiple meta-learner configurations to stack our 6 model prediction probability matrices (18 features total) using 5-fold Stratified CV:
1. **Logistic Regression (Baseline):** Multinomial linear classifier.
2. **ElasticNet Stacker:** `LogisticRegression` with SAGA solver, `penalty='elasticnet'`, `l1_ratio=0.5`.
3. **Shallow LightGBM Meta-Learner:** Highly regularized shallow GBDT classifier.
4. **Shallow XGBoost Meta-Learner:** Highly regularized shallow GBDT classifier.
5. **Direct Nelder-Mead Blender (`nm_blend`):** Optimization searching directly for optimal blending weights $(w_1, \dots, w_6)$ to maximize Balanced Accuracy on OOF validation.
6. **Calibrated Stackers:** We apply Platt Scaling and Isotonic Regression fold-by-fold to configurations 1-4.

The best configuration is selected dynamically based on out-of-fold Balanced Accuracy."""
    cells[stacking_md_idx]["source"] = format_source(stacking_md_source)
    
    # 3. Update stacking_meta_code
    stacking_code_idx = find_cell_by_id(cells, "stacking_meta_code")
    stacking_code_source = r"""# Helper functions for probability calibration on a specific fold split
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

# Initialize predictions arrays for all configurations
lr_oof = np.zeros((len(y), 3))
mlp_oof = np.zeros((len(y), 3))
elasticnet_oof = np.zeros((len(y), 3))
lgb_meta_oof = np.zeros((len(y), 3))
xgb_meta_oof = np.zeros((len(y), 3))
nm_blend_oof = np.zeros((len(y), 3))

# Calibrated variants
lr_platt_oof = np.zeros((len(y), 3))
mlp_platt_oof = np.zeros((len(y), 3))
elasticnet_platt_oof = np.zeros((len(y), 3))
lgb_meta_platt_oof = np.zeros((len(y), 3))
xgb_meta_platt_oof = np.zeros((len(y), 3))

lr_iso_oof = np.zeros((len(y), 3))
mlp_iso_oof = np.zeros((len(y), 3))
elasticnet_iso_oof = np.zeros((len(y), 3))
lgb_meta_iso_oof = np.zeros((len(y), 3))
xgb_meta_iso_oof = np.zeros((len(y), 3))

# Define classifiers
lr_meta = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=42)
mlp_meta = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=0.01, early_stopping=True, validation_fraction=0.1, random_state=42, max_iter=500)
elasticnet_meta = LogisticRegression(max_iter=2000, penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1, class_weight='balanced', random_state=42)
lgb_meta_model = lgb.LGBMClassifier(max_depth=2, n_estimators=100, learning_rate=0.03, reg_alpha=5.0, reg_lambda=5.0, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)
xgb_meta_model = xgb.XGBClassifier(max_depth=2, n_estimators=100, learning_rate=0.03, reg_alpha=5.0, reg_lambda=5.0, random_state=42, n_jobs=-1, eval_metric='mlogloss')

def optimize_blend_weights(oof_train, y_train):
    def loss_func(weights):
        w = np.clip(weights, 0, None)
        if w.sum() == 0:
            return 999.0
        w = w / w.sum()
        
        blend_probs = np.zeros((len(y_train), 3))
        for i in range(6):
            blend_probs += w[i] * oof_train[:, i*3:(i+1)*3]
        preds = np.argmax(blend_probs, axis=1)
        return -balanced_accuracy_score(y_train, preds)
        
    init_weights = [1.0 / 6] * 6
    res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 200})
    best_w = np.clip(res.x, 0, None)
    if best_w.sum() == 0:
        best_w = np.ones(6) / 6.0
    else:
        best_w = best_w / best_w.sum()
    return best_w

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("Evaluating Stacker Configurations via Nested 5-Fold Stratified CV...")
for fold, (train_idx, val_idx) in enumerate(skf.split(Xoof, y), 1):
    X_tr, y_tr = Xoof[train_idx], y[train_idx]
    X_va, y_va = Xoof[val_idx], y[val_idx]
    
    # 1. Raw stackers
    lr_meta.fit(X_tr, y_tr)
    lr_oof[val_idx] = lr_meta.predict_proba(X_va)
    
    mlp_meta.fit(X_tr, y_tr)
    mlp_oof[val_idx] = mlp_meta.predict_proba(X_va)
    
    elasticnet_meta.fit(X_tr, y_tr)
    elasticnet_oof[val_idx] = elasticnet_meta.predict_proba(X_va)
    
    lgb_meta_model.fit(X_tr, y_tr)
    lgb_meta_oof[val_idx] = lgb_meta_model.predict_proba(X_va)
    
    xgb_meta_model.fit(X_tr, y_tr)
    xgb_meta_oof[val_idx] = xgb_meta_model.predict_proba(X_va)
    
    # Nelder-Mead direct blend
    best_w = optimize_blend_weights(X_tr, y_tr)
    fold_blend = np.zeros((len(val_idx), 3))
    for i in range(6):
        fold_blend += best_w[i] * X_va[:, i*3:(i+1)*3]
    nm_blend_oof[val_idx] = fold_blend
    
    # 2. Platt calibrated stackers
    X_tr_platt, X_va_platt = calibrate_platt_split(X_tr, X_va, y_tr)
    
    lr_meta.fit(X_tr_platt, y_tr)
    lr_platt_oof[val_idx] = lr_meta.predict_proba(X_va_platt)
    
    mlp_meta.fit(X_tr_platt, y_tr)
    mlp_platt_oof[val_idx] = mlp_meta.predict_proba(X_va_platt)
    
    elasticnet_meta.fit(X_tr_platt, y_tr)
    elasticnet_platt_oof[val_idx] = elasticnet_meta.predict_proba(X_va_platt)
    
    lgb_meta_model.fit(X_tr_platt, y_tr)
    lgb_meta_platt_oof[val_idx] = lgb_meta_model.predict_proba(X_va_platt)
    
    xgb_meta_model.fit(X_tr_platt, y_tr)
    xgb_meta_platt_oof[val_idx] = xgb_meta_model.predict_proba(X_va_platt)
    
    # 3. Isotonic calibrated stackers
    X_tr_iso, X_va_iso = calibrate_isotonic_split(X_tr, X_va, y_tr)
    
    lr_meta.fit(X_tr_iso, y_tr)
    lr_iso_oof[val_idx] = lr_meta.predict_proba(X_va_iso)
    
    mlp_meta.fit(X_tr_iso, y_tr)
    mlp_iso_oof[val_idx] = mlp_meta.predict_proba(X_va_iso)
    
    elasticnet_meta.fit(X_tr_iso, y_tr)
    elasticnet_iso_oof[val_idx] = elasticnet_meta.predict_proba(X_va_iso)
    
    lgb_meta_model.fit(X_tr_iso, y_tr)
    lgb_meta_iso_oof[val_idx] = lgb_meta_model.predict_proba(X_va_iso)
    
    xgb_meta_model.fit(X_tr_iso, y_tr)
    xgb_meta_iso_oof[val_idx] = xgb_meta_model.predict_proba(X_va_iso)

# Compute validation scores
scores = {
    'lr': (balanced_accuracy_score(y, np.argmax(lr_oof, axis=1)), lr_oof),
    'mlp': (balanced_accuracy_score(y, np.argmax(mlp_oof, axis=1)), mlp_oof),
    'elasticnet': (balanced_accuracy_score(y, np.argmax(elasticnet_oof, axis=1)), elasticnet_oof),
    'lgb_meta': (balanced_accuracy_score(y, np.argmax(lgb_meta_oof, axis=1)), lgb_meta_oof),
    'xgb_meta': (balanced_accuracy_score(y, np.argmax(xgb_meta_oof, axis=1)), xgb_meta_oof),
    'nm_blend': (balanced_accuracy_score(y, np.argmax(nm_blend_oof, axis=1)), nm_blend_oof),
    
    'lr_platt': (balanced_accuracy_score(y, np.argmax(lr_platt_oof, axis=1)), lr_platt_oof),
    'mlp_platt': (balanced_accuracy_score(y, np.argmax(mlp_platt_oof, axis=1)), mlp_platt_oof),
    'elasticnet_platt': (balanced_accuracy_score(y, np.argmax(elasticnet_platt_oof, axis=1)), elasticnet_platt_oof),
    'lgb_meta_platt': (balanced_accuracy_score(y, np.argmax(lgb_meta_platt_oof, axis=1)), lgb_meta_platt_oof),
    'xgb_meta_platt': (balanced_accuracy_score(y, np.argmax(xgb_meta_platt_oof, axis=1)), xgb_meta_platt_oof),
    
    'lr_iso': (balanced_accuracy_score(y, np.argmax(lr_iso_oof, axis=1)), lr_iso_oof),
    'mlp_iso': (balanced_accuracy_score(y, np.argmax(mlp_iso_oof, axis=1)), mlp_iso_oof),
    'elasticnet_iso': (balanced_accuracy_score(y, np.argmax(elasticnet_iso_oof, axis=1)), elasticnet_iso_oof),
    'lgb_meta_iso': (balanced_accuracy_score(y, np.argmax(lgb_meta_iso_oof, axis=1)), lgb_meta_iso_oof),
    'xgb_meta_iso': (balanced_accuracy_score(y, np.argmax(xgb_meta_iso_oof, axis=1)), xgb_meta_iso_oof)
}

print("\n--- Out-of-Fold Balanced Accuracy Scores ---")
for name, (score, _) in scores.items():
    print(f"{name.upper():<20}: {score:.5f}")

best_name = max(scores, key=lambda k: scores[k][0])
best_score, best_oof = scores[best_name]
print(f"\n=== Dynamic Selection: Best Meta-Learner configuration is '{best_name}' ===")
print(f"Best OOF Balanced Accuracy Score: {best_score:.5f}")

# --- Generate Final Test Predictions using full dataset ---
if best_name == 'nm_blend':
    best_w = optimize_blend_weights(Xoof, y)
    print(f"Optimal Nelder-Mead Blending Weights: {best_w}")
    stacked_test_prob = np.zeros((len(Xtest), 3))
    for i in range(6):
        stacked_test_prob += best_w[i] * Xtest[:, i*3:(i+1)*3]
elif 'platt' in best_name:
    best_train, best_test = calibrate_platt_split(Xoof, Xtest, y)
    if 'mlp' in best_name:
        best_model = mlp_meta
    elif 'elasticnet' in best_name:
        best_model = elasticnet_meta
    elif 'lgb_meta' in best_name:
        best_model = lgb_meta_model
    elif 'xgb_meta' in best_name:
        best_model = xgb_meta_model
    else:
        best_model = lr_meta
    best_model.fit(best_train, y)
    stacked_test_prob = best_model.predict_proba(best_test)
elif 'iso' in best_name:
    best_train, best_test = calibrate_isotonic_split(Xoof, Xtest, y)
    if 'mlp' in best_name:
        best_model = mlp_meta
    elif 'elasticnet' in best_name:
        best_model = elasticnet_meta
    elif 'lgb_meta' in best_name:
        best_model = lgb_meta_model
    elif 'xgb_meta' in best_name:
        best_model = xgb_meta_model
    else:
        best_model = lr_meta
    best_model.fit(best_train, y)
    stacked_test_prob = best_model.predict_proba(best_test)
else:
    if 'mlp' in best_name:
        best_model = mlp_meta
    elif 'elasticnet' in best_name:
        best_model = elasticnet_meta
    elif 'lgb_meta' in best_name:
        best_model = lgb_meta_model
    elif 'xgb_meta' in best_name:
        best_model = xgb_meta_model
    else:
        best_model = lr_meta
    best_model.fit(Xoof, y)
    stacked_test_prob = best_model.predict_proba(Xtest)

meta_oof = best_oof
stacked_score = best_score"""
    cells[stacking_code_idx]["source"] = format_source(stacking_code_source)
    
    # Save the updated notebook
    print(f"Saving {path}...")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
        
    print("Update complete!")

if __name__ == '__main__':
    update_blender()
