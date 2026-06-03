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

def update_ensemble_blender_v6():
    notebook_path = "notebooks/03_model_tuning_and_ensemble.ipynb"
    
    # Idempotency Git Checkout: Check if inside git repository before executing git checkout
    is_git = os.system("git rev-parse --is-inside-work-tree >/dev/null 2>&1") == 0
    if is_git:
        print("Restoring notebook from git to ensure a clean starting point...")
        os.system(f"git checkout HEAD -- {notebook_path}")
    else:
        print("Not inside a git repository or git command not available. Skipping git checkout...")
    
    print(f"Reading {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    cells = notebook["cells"]
    print(f"Initial cells count: {len(cells)}")
    
    # Cell 0 (Title & Intro)
    cell_0_source = r"""# Stellar Classification: Stacked Ensembling & Stacking-Assisted Blender

This notebook executes prediction ensembling by combining a class-balanced Logistic Regression stacking meta-learner with high-scoring external submissions using a hybrid voting blender. Decision boundaries are tuned using Nelder-Mead optimization on out-of-fold predictions."""
    cells[0]["source"] = format_source(cell_0_source)
    
    # Cell 1 (Imports)
    cell_1_source = r"""import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import os
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from scipy.optimize import minimize

warnings.filterwarnings('ignore', category=FutureWarning)

# Target mapping variables
TARGET_MAP = {'GALAXY': 0, 'QSO': 1, 'STAR': 2}
INV_TARGET_MAP = {0: 'GALAXY', 1: 'QSO', 2: 'STAR'}"""
    cells[1]["source"] = format_source(cell_1_source)
    
    # Cell 3 (Path Configs)
    cell_3_source = r"""# Dynamic path resolution for Local vs Kaggle environment
if os.path.exists('/kaggle/input/competitions/playground-series-s6e6'):
    DATA_DIR = '/kaggle/input/competitions/playground-series-s6e6'
    PRED_DIR = '/kaggle/input/notebooks/tuannm3812/stellar-classification-baseline-modeling/predictions'
    SUBS_DIR = Path('/kaggle/input/stellar-data/external/submissions')
    OUTPUT_DIR = '.'
else:
    DATA_DIR = '../data'
    PRED_DIR = '../predictions'
    SUBS_DIR = Path('../scratch/external/submissions')
    OUTPUT_DIR = '..'

print(f"Data Directory: {DATA_DIR}")
print(f"Predictions Directory: {PRED_DIR}")
print(f"Submissions Directory: {SUBS_DIR}")
print(f"Output Directory: {OUTPUT_DIR}")"""
    cells[3]["source"] = format_source(cell_3_source)
    
    # Injected disagreement cells (placed after cell 7)
    disagreement_md = {
        "cell_type": "markdown",
        "id": "disagreement_analysis_md",
        "metadata": {},
        "source": format_source(r"""### 3.2. Model Disagreement Analysis
Ensembling is only effective if the component models make *different* errors. We isolate the disagreement region—rows where our models do not predict the same class—and print the agreement statistics. 

We also load the 5 external high-scoring submissions to analyze their disagreement region.""")
    }
    
    disagreement_code = {
        "cell_type": "code",
        "execution_count": None,
        "id": "disagreement_analysis_code",
        "metadata": {},
        "outputs": [],
        "source": format_source(r"""# Convert probabilities to hard class labels
lgb_preds = np.argmax(lgb_oof, axis=1)
xgb_preds = np.argmax(xgb_oof, axis=1)
cat_preds = np.argmax(cat_oof, axis=1)

preds_matrix = np.column_stack([lgb_preds, xgb_preds, cat_preds])
model_names = ['LightGBM', 'XGBoost', 'CatBoost']

# Calculate disagreement rows
unanimous = np.all(preds_matrix == preds_matrix[:, [0]], axis=1)
disagree = ~unanimous
print(f"Total training rows:  {len(y)}")
print(f"Unanimous agreement:  {unanimous.sum()} ({unanimous.mean()*100:.2f}%)")
print(f"Disagreement region:  {disagree.sum()} ({disagree.mean()*100:.2f}%)\n")

# Pairwise agreement matrix
agree = np.zeros((3, 3))
for a in range(3):
    for b in range(3):
        agree[a, b] = (preds_matrix[:, a] == preds_matrix[:, b]).mean() * 100

plt.figure(figsize=(6, 5))
sns.heatmap(pd.DataFrame(agree, index=model_names, columns=model_names), 
            annot=True, fmt='.2f', cmap='viridis', square=True, cbar_kws={'label': 'Agreement %'})
plt.title('Pairwise Model Agreement Heatmap')
plt.tight_layout()
plt.show()

# Load external submissions if available
sub_files = sorted(SUBS_DIR.glob('*.csv')) if SUBS_DIR.exists() else []
def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False
sub_files = [f for f in sub_files if is_float(f.stem)]

if sub_files:
    print(f"Loaded {len(sub_files)} external submissions from {SUBS_DIR}:")
    external_scores = {f.stem: float(f.stem) for f in sub_files}
    external_subs = {f.stem: pd.read_csv(f).sort_values('id').reset_index(drop=True) for f in sub_files}
    
    # Verify ID alignment
    ref_id = external_subs[sub_files[0].stem]['id'].values
    for name, df in external_subs.items():
        assert np.array_equal(df['id'].values, ref_id), f"{name}: id mismatch"
        
    L = np.column_stack([external_subs[name]['class'].map(TARGET_MAP).values for name in external_scores])
    W = np.array([external_scores[name] for name in external_scores])
    
    unanimous_ext = np.all(L == L[:, [0]], axis=1)
    disagree_ext = ~unanimous_ext
    print(f"Total test rows:  {len(ref_id)}")
    print(f"Unanimous external agreement:  {unanimous_ext.sum()} ({unanimous_ext.mean()*100:.2f}%)")
    print(f"External disagreement region:  {disagree_ext.sum()} ({disagree_ext.mean()*100:.2f}%)")
else:
    print("External submissions directory not found. Please mount flexonafft/stellar-data.")""")
    }
    
    # Replace cell 8, 9, 10, 11, 12, 13
    cell_8_source = r"""## 4. Stacking & Threshold Tuning

### 4.1. Logistic Regression Stacking
Instead of manual grid-search probability blending, we implement a **Logistic Regression meta-learner** to stack model probabilities. We train the meta-model on OOF prediction vectors using 5-fold Stratified CV to predict target classes, then fit it on the full dataset to generate final test probabilities."""
    
    cell_9_source = r"""# Stack predictions: shape (N, 3 * 3 = 9)
Xoof = np.hstack([lgb_oof, xgb_oof, cat_oof])
Xtest = np.hstack([lgb_test, xgb_test, cat_test])

# Initialize Logistic Regression Meta-Learner
meta_model = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Generate leakage-free OOF stacked predictions
print("--- Training Stacking Meta-Learner ---")
meta_oof = cross_val_predict(meta_model, Xoof, y, cv=skf, method='predict_proba')

# Evaluate Meta-Learner OOF Balanced Accuracy
stacked_score = balanced_accuracy_score(y, np.argmax(meta_oof, axis=1))
print(f"Stacked Meta-Learner OOF Balanced Accuracy: {stacked_score:.5f}")

# Fit on full OOF predictions to generate test probabilities
meta_model.fit(Xoof, y)
stacked_test_prob = meta_model.predict_proba(Xtest)"""
    
    cell_10_source = r"""### 4.2. Nelder-Mead Threshold Optimization
Standard inference assigns classes using $\arg\max(P_c)$. However, when optimizing for Balanced Accuracy (unweighted average recall) under severe class skew, the standard probability boundaries are mathematically sub-optimal. We search for class-specific multipliers using the **Nelder-Mead simplex algorithm** (via `scipy.optimize.minimize`) to directly maximize Balanced Accuracy on our Out-of-Fold (OOF) predictions."""
    
    cell_11_source = r"""def optimize_thresholds(oof_probs, y_true):
    def loss_func(weights):
        if np.any(weights <= 0.01):
            return 999.0 + np.sum(np.abs(weights))
        scaled_probs = oof_probs * weights
        preds = np.argmax(scaled_probs, axis=1)
        return -balanced_accuracy_score(y_true, preds)
    
    init_weights = [1.0, 1.0, 1.0]
    res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 500})
    best_weights = res.x / np.sum(res.x) * 3.0
    best_score = -res.fun
    return best_weights, best_score, res.success, res.message

print("--- Optimizing Decision Thresholds over Stacked Predictions ---")
best_thresholds, optimized_score, opt_success, opt_msg = optimize_thresholds(meta_oof, y)
if not opt_success:
    print(f"Warning: Nelder-Mead threshold optimization did not converge: {opt_msg}")

t_gal, t_qso, t_star = best_thresholds

print(f"Original Stacking Score: {stacked_score:.5f}")
print(f"Optimized Stacking Score: {optimized_score:.5f}")
print(f"Optimal Multipliers -> GALAXY: {t_gal:.4f}, QSO: {t_qso:.4f}, STAR: {t_star:.4f}")"""
    
    cell_12_source = r"""## 5. Final Inference & Submission

### 5.1. Stacking-Assisted Blender
We combine our stacked probabilities with 5 external high-scoring submissions. Unanimous rows are kept; disagreement rows are decided by the sum of external weights plus our stacked probabilities weighted by the panel sum, scaled by tuned decision thresholds.

### 5.2. Generate Submission File
We save the class predictions to `submission.csv` using the inverse target mapping, formatting the output for final leaderboard submission."""
    
    cell_13_source = r"""# Stacking-Assisted Blender Inference

# 1. Load external submissions
sub_files = sorted(SUBS_DIR.glob('*.csv')) if SUBS_DIR.exists() else []
def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False
sub_files = [f for f in sub_files if is_float(f.stem)]

if not sub_files:
    print("Warning: No external submissions found. Generating submission using local Stacked Meta-Learner + thresholds.")
    sub_sample_path = os.path.join(DATA_DIR, 'sample_submission.csv')
    submission = pd.read_csv(sub_sample_path)
    ref_id = submission['id'].values
    
    # Scale local probabilities and classify
    scaled_probs = stacked_test_prob * best_thresholds
    final_preds = np.argmax(scaled_probs, axis=1)
else:
    external_scores = {f.stem: float(f.stem) for f in sub_files}
    external_subs = {f.stem: pd.read_csv(f).sort_values('id').reset_index(drop=True) for f in sub_files}

    # Verify ID and class label validity across all external submissions before ensembling
    ref_id = external_subs[sub_files[0].stem]['id'].values
    for name, df in external_subs.items():
        assert np.array_equal(df['id'].values, ref_id), f"{name}: id mismatch"
        invalid_classes = df[~df['class'].isin(TARGET_MAP.keys())]['class'].unique()
        assert len(invalid_classes) == 0, f"{name} contains invalid classes: {invalid_classes}"

    L = np.column_stack([external_subs[name]['class'].map(TARGET_MAP).values for name in external_scores])
    W = np.array([external_scores[name] for name in external_scores])
    
    # Cast class indices to integers for numpy indexing safety
    L = L.astype(int)

    # 2. Compute soft votes and merge layers
    CORE_WEIGHT = float(W.sum())
    n_samples = len(ref_id)
    votes = np.zeros((n_samples, 3), dtype=np.float64)

    # Accumulate hard votes from external submissions
    for j in range(len(sub_files)):
        np.add.at(votes, (np.arange(n_samples), L[:, j]), W[j])

    # Calibrate and normalize stacked test probabilities
    calibrated_stacked_prob = stacked_test_prob * best_thresholds
    normalized_prob = calibrated_stacked_prob / np.sum(calibrated_stacked_prob, axis=1, keepdims=True)
    
    # Accumulate soft votes from stacked test probabilities scaled by CORE_WEIGHT
    votes += CORE_WEIGHT * normalized_prob

    final_preds = np.argmax(votes, axis=1)

    # Enforce consensus on unanimous rows
    unanimous_ext = np.all(L == L[:, [0]], axis=1)
    final_preds[unanimous_ext] = L[unanimous_ext, 0]
    
    submission = pd.DataFrame({'id': ref_id})

# Create submission DataFrame class column
submission['class'] = pd.Series(final_preds).map(INV_TARGET_MAP)

# Save to CSV
sub_out_path = os.path.join(OUTPUT_DIR, 'submission.csv')
submission.to_csv(sub_out_path, index=False)
print(f"Submission file created successfully at {sub_out_path}!")
print(submission['class'].value_counts())"""
    
    new_cells = []
    # Cells 0 to 7
    for i in range(8):
        new_cells.append(cells[i].copy())
    
    # Cell 8: disagreement markdown
    new_cells.append(disagreement_md)
    
    # Cell 9: disagreement code
    new_cells.append(disagreement_code)
    
    # Cell 10: Stacking markdown (copied from cells[8] and updated)
    cell_10 = cells[8].copy()
    cell_10["source"] = format_source(cell_8_source)
    new_cells.append(cell_10)
    
    # Cell 11: Stacking code (copied from cells[9] and updated)
    cell_11 = cells[9].copy()
    cell_11["source"] = format_source(cell_9_source)
    cell_11["outputs"] = []
    cell_11["execution_count"] = None
    new_cells.append(cell_11)
    
    # Cell 12: Nelder-Mead markdown (copied from cells[10] and updated)
    cell_12 = cells[10].copy()
    cell_12["source"] = format_source(cell_10_source)
    new_cells.append(cell_12)
    
    # Cell 13: Nelder-Mead code (copied from cells[11] and updated)
    cell_13 = cells[11].copy()
    cell_13["source"] = format_source(cell_11_source)
    cell_13["outputs"] = []
    cell_13["execution_count"] = None
    new_cells.append(cell_13)
    
    # Cell 14: final inference markdown (copied from cells[12] and updated)
    cell_14 = cells[12].copy()
    cell_14["source"] = format_source(cell_12_source)
    new_cells.append(cell_14)
    
    # Cell 15: final inference code (copied from cells[13] and updated)
    cell_15 = cells[13].copy()
    cell_15["source"] = format_source(cell_13_source)
    cell_15["outputs"] = []
    cell_15["execution_count"] = None
    new_cells.append(cell_15)
    
    # Assign unique IDs to all 16 cells to avoid nbformat duplication errors
    cell_ids = [
        "title_md", "imports_code", "configs_md", "configs_code",
        "load_preds_md", "load_preds_code", "baseline_eval_md", "baseline_eval_code",
        "disagreement_analysis_md", "disagreement_analysis_code",
        "stacking_meta_md", "stacking_meta_code",
        "threshold_tuning_md", "threshold_tuning_code",
        "inference_blender_md", "inference_blender_code"
    ]
    for idx, cell in enumerate(new_cells):
        cell["id"] = cell_ids[idx]
        
    notebook["cells"] = new_cells
    print(f"Final cells count: {len(notebook['cells'])}")
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Successfully updated {notebook_path}!")

if __name__ == "__main__":
    update_ensemble_blender_v6()
