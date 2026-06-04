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

def update_ensemble():
    path = "notebooks/03_model_tuning_and_ensemble.ipynb"
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    cells = data["cells"]
    
    # 1. Update loading probabilities cell (cell index 5) to include new models
    cell_5_source = r"""# Load ground truth class labels
y_train_path = os.path.join(DATA_DIR, 'train.csv')
y = pd.read_csv(y_train_path)['class'].map(TARGET_MAP).values

# Load OOF prediction probability files (baseline models)
lgb_oof = np.load(os.path.join(PRED_DIR, 'lgb_oof.npy'))
xgb_oof = np.load(os.path.join(PRED_DIR, 'xgb_oof.npy'))
cat_oof = np.load(os.path.join(PRED_DIR, 'cat_oof.npy'))
rf_oof = np.load(os.path.join(PRED_DIR, 'rf_oof.npy'))
et_oof = np.load(os.path.join(PRED_DIR, 'et_oof.npy'))
hist_gb_oof = np.load(os.path.join(PRED_DIR, 'hist_gb_oof.npy'))

# Load Test prediction probability files
lgb_test = np.load(os.path.join(PRED_DIR, 'lgb_test.npy'))
xgb_test = np.load(os.path.join(PRED_DIR, 'xgb_test.npy'))
cat_test = np.load(os.path.join(PRED_DIR, 'cat_test.npy'))
rf_test = np.load(os.path.join(PRED_DIR, 'rf_test.npy'))
et_test = np.load(os.path.join(PRED_DIR, 'et_test.npy'))
hist_gb_test = np.load(os.path.join(PRED_DIR, 'hist_gb_test.npy'))

print(f"Loaded shapes - y: {y.shape}")
print(f"OOF Shapes: LGB={lgb_oof.shape}, XGB={xgb_oof.shape}, CAT={cat_oof.shape}, RF={rf_oof.shape}, ET={et_oof.shape}, HIST={hist_gb_oof.shape}")"""
    cells[5]["source"] = format_source(cell_5_source)
    
    # 2. Update individual performance cell (cell index 7) to include new models
    cell_7_source = r"""lgb_score = balanced_accuracy_score(y, np.argmax(lgb_oof, axis=1))
xgb_score = balanced_accuracy_score(y, np.argmax(xgb_oof, axis=1))
cat_score = balanced_accuracy_score(y, np.argmax(cat_oof, axis=1))
rf_score = balanced_accuracy_score(y, np.argmax(rf_oof, axis=1))
et_score = balanced_accuracy_score(y, np.argmax(et_oof, axis=1))
hist_gb_score = balanced_accuracy_score(y, np.argmax(hist_gb_oof, axis=1))

print("Individual Model Scores (Balanced Accuracy):")
print(f"LightGBM: {lgb_score:.5f}")
print(f"XGBoost:  {xgb_score:.5f}")
print(f"CatBoost: {cat_score:.5f}")
print(f"RF:       {rf_score:.5f}")
print(f"ET:       {et_score:.5f}")
print(f"HistGB:   {hist_gb_score:.5f}")"""
    cells[7]["source"] = format_source(cell_7_source)
    
    # 3. Update disagreement analysis code (cell index 9) to include all 6 models
    cell_9_source = r"""import matplotlib.pyplot as plt
import seaborn as sns

# Convert probabilities to hard class labels
lgb_preds = np.argmax(lgb_oof, axis=1)
xgb_preds = np.argmax(xgb_oof, axis=1)
cat_preds = np.argmax(cat_oof, axis=1)
rf_preds = np.argmax(rf_oof, axis=1)
et_preds = np.argmax(et_oof, axis=1)
hist_gb_preds = np.argmax(hist_gb_oof, axis=1)

preds_matrix = np.column_stack([lgb_preds, xgb_preds, cat_preds, rf_preds, et_preds, hist_gb_preds])
model_names = ['LightGBM', 'XGBoost', 'CatBoost', 'RF', 'ET', 'HistGB']

# Calculate disagreement rows
unanimous = np.all(preds_matrix == preds_matrix[:, [0]], axis=1)
disagree = ~unanimous
print(f"Total training rows:  {len(y)}")
print(f"Unanimous agreement:  {unanimous.sum()} ({unanimous.mean()*100:.2f}%)")
print(f"Disagreement region:  {disagree.sum()} ({disagree.mean()*100:.2f}%)\n")

# Pairwise agreement matrix
agree = np.zeros((6, 6))
for a in range(6):
    for b in range(6):
        agree[a, b] = (preds_matrix[:, a] == preds_matrix[:, b]).mean() * 100

plt.figure(figsize=(8, 7))
sns.heatmap(pd.DataFrame(agree, index=model_names, columns=model_names), 
            annot=True, fmt='.2f', cmap='viridis', square=True, cbar_kws={'label': 'Agreement %'})
plt.title('Pairwise Model Agreement Heatmap')
plt.tight_layout()
plt.show()

# Load external submissions if available
sub_files = sorted(SUBS_DIR.glob('*.csv')) if SUBS_DIR.exists() else []
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
    print("External submissions directory not found. Please mount flexonafft/stellar-data.")"""
    cells[9]["source"] = format_source(cell_9_source)
    
    # 4. Update Logistic Regression Stacking (cell index 11) to hstack all 6 models
    cell_11_source = r"""from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold

# Stack predictions: shape (N, 6 * 3 = 18)
Xoof = np.hstack([lgb_oof, xgb_oof, cat_oof, rf_oof, et_oof, hist_gb_oof])
Xtest = np.hstack([lgb_test, xgb_test, cat_test, rf_test, et_test, hist_gb_test])

# Initialize Logistic Regression Meta-Learner
meta_model = LogisticRegression(max_iter=2000, C=1.0, multi_class='multinomial', class_weight='balanced', random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Generate leakage-free OOF stacked predictions
print("Training Stacking Meta-Learner...")
meta_oof = cross_val_predict(meta_model, Xoof, y, cv=skf, method='predict_proba')

# Evaluate Meta-Learner OOF Balanced Accuracy
stacked_score = balanced_accuracy_score(y, np.argmax(meta_oof, axis=1))
print(f"Stacked Meta-Learner OOF Balanced Accuracy: {stacked_score:.5f}")

# Fit on full OOF predictions to generate test probabilities
meta_model.fit(Xoof, y)
stacked_test_prob = meta_model.predict_proba(Xtest)"""
    cells[11]["source"] = format_source(cell_11_source)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(f"Successfully updated {path}!")

if __name__ == "__main__":
    update_ensemble()
