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

def update_ensemble_stacking_v5():
    notebook_path = "notebooks/03_model_tuning_and_ensemble.ipynb"
    
    # 0. Ensure idempotency by restoring the notebook from git before reading
    print("Restoring notebook from git to ensure a clean starting point...")
    os.system(f"git checkout {notebook_path}")
    
    print(f"Reading {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
    
    cells = notebook["cells"]
    print(f"Initial cells count: {len(cells)}")
    
    # 1. Update Cell 0 markdown (Title & Intro)
    cell_0_source = r"""# Stellar Classification: Stacked Ensembling & Threshold Tuning

This notebook executes prediction stacking using a Logistic Regression meta-learner and Nelder-Mead decision threshold optimization across LightGBM, XGBoost, and CatBoost predictions. This maximizes the cross-validation Balanced Accuracy score and generates the final predictions."""
    cells[0]["source"] = format_source(cell_0_source)
    
    # 2. Update Cell 2 markdown
    cell_2_source = r"""## 1. Directory Configurations & Dynamic Outputs

We configure dataset directories, prediction probability sources, and target output paths, ensuring portability across local and Kaggle environments."""
    cells[2]["source"] = format_source(cell_2_source)
    
    # 3. Update Cell 4 markdown
    cell_4_source = r"""## 2. Loading Probability Predictions & Target Labels

We load the ground-truth training classes alongside the saved out-of-fold validation and test prediction probability matrices. Stacking prediction probabilities (confidence scores) rather than discrete labels maintains calibration and improves class boundary separation."""
    cells[4]["source"] = format_source(cell_4_source)
    
    # 4. Modify Cell 6 markdown (Individual Performance Review)
    cell_6_source = r"""## 3. Evaluate Baseline Models

### 3.1. Individual Model Scores
We verify the baseline Balanced Accuracy scores of the individual models before ensembling. This establishes the baseline performance threshold that our ensemble optimization must exceed."""
    cells[6]["source"] = format_source(cell_6_source)
    
    # 5. Keep Cell 7 as is.
    
    # 6. Inject two new cells right after cell 7
    disagreement_md = {
        "cell_type": "markdown",
        "id": "disagreement_analysis_md",
        "metadata": {},
        "source": format_source(r"""### 3.2. Model Disagreement Analysis
Ensembling is only effective if the component models make *different* errors. We isolate the disagreement region—rows where our models do not predict the same class—and print the pairwise agreement percentages.""")
    }
    
    disagreement_code = {
        "cell_type": "code",
        "execution_count": None,
        "id": "disagreement_analysis_code",
        "metadata": {},
        "outputs": [],
        "source": format_source(r"""import matplotlib.pyplot as plt
import seaborn as sns

# Convert probabilities to hard class labels
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
plt.show()""")
    }
    
    # 7. Replace old cell 8, 9, 10, 11, 12, 13
    cell_8_source = r"""## 4. Stacking & Threshold Tuning

### 4.1. Logistic Regression Stacking
Instead of manual grid-search probability blending, we implement a **Logistic Regression meta-learner** to stack model probabilities. We train the meta-model on OOF prediction vectors using 5-fold Stratified CV to predict target classes, then fit it on the full dataset to generate final test probabilities."""
    
    cell_9_source = r"""from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Stack predictions: shape (N, 3 * 3 = 9)
Xoof = np.hstack([lgb_oof, xgb_oof, cat_oof])
Xtest = np.hstack([lgb_test, xgb_test, cat_test])

# Initialize Logistic Regression Meta-Learner
meta_model = LogisticRegression(max_iter=2000, C=1.0, multi_class='multinomial', class_weight='balanced', random_state=42)
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
    
    cell_11_source = r"""from scipy.optimize import minimize

def optimize_thresholds(oof_probs, y_true):
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

### 5.1. Test Class Scaling
We apply the optimized threshold multipliers to the stacked test probabilities from the Logistic Regression meta-learner prior to argmax classification.

### 5.2. Generate Submission File
We save the class predictions to `submission.csv` using the inverse target mapping, formatting the output for final leaderboard submission."""
    
    cell_13_source = r"""# Applying stacked test predictions from Logistic Regression meta-learner
final_test_prob = stacked_test_prob

# Applying optimized threshold multipliers before taking argmax
scaled_test_prob = final_test_prob * best_thresholds
final_preds = np.argmax(scaled_test_prob, axis=1)

# Create submission DataFrame
sub_sample_path = os.path.join(DATA_DIR, 'sample_submission.csv')
submission = pd.read_csv(sub_sample_path)
submission['class'] = pd.Series(final_preds).map(INV_TARGET_MAP)

# Save to CSV
sub_out_path = os.path.join(OUTPUT_DIR, 'submission.csv')
submission.to_csv(sub_out_path, index=False)
print(f"Submission file created successfully at {sub_out_path}!")
print(submission['class'].value_counts())"""
    
    # We construct the new cells array step-by-step
    new_cells = []
    # Cells 0 to 7
    for i in range(8):
        new_cells.append(cells[i])
    
    # Inject disagreement cells
    new_cells.append(disagreement_md)
    new_cells.append(disagreement_code)
    
    # Cells 8 to 13 are replaced
    # Cell 8
    cell_8 = cells[8].copy()
    cell_8["source"] = format_source(cell_8_source)
    new_cells.append(cell_8)
    
    # Cell 9
    cell_9 = cells[9].copy()
    cell_9["source"] = format_source(cell_9_source)
    cell_9["outputs"] = []
    cell_9["execution_count"] = None
    new_cells.append(cell_9)
    
    # Cell 10
    cell_10 = cells[10].copy()
    cell_10["source"] = format_source(cell_10_source)
    new_cells.append(cell_10)
    
    # Cell 11
    cell_11 = cells[11].copy()
    cell_11["source"] = format_source(cell_11_source)
    cell_11["outputs"] = []
    cell_11["execution_count"] = None
    new_cells.append(cell_11)
    
    # Cell 12
    cell_12 = cells[12].copy()
    cell_12["source"] = format_source(cell_12_source)
    new_cells.append(cell_12)
    
    # Cell 13
    cell_13 = cells[13].copy()
    cell_13["source"] = format_source(cell_13_source)
    cell_13["outputs"] = []
    cell_13["execution_count"] = None
    new_cells.append(cell_13)
    
    # Copy any remaining cells from original notebook to keep execution idempotent and non-destructive
    if len(cells) > 14:
        print(f"Appending {len(cells) - 14} remaining cells...")
        new_cells.extend(cells[14:])
    
    # Update notebook cells list
    notebook["cells"] = new_cells
    print(f"Final cells count: {len(notebook['cells'])}")
    
    # Write updated notebook
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"Successfully updated {notebook_path}!")

if __name__ == "__main__":
    update_ensemble_stacking_v5()
