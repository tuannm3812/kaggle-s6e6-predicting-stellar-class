# Advanced Stacking Meta-Learners Design Specification

## Goal
Implement a wider suite of ensembling meta-learners—incorporating ElasticNet regularisation, regularised shallow GBDT classifiers, and direct Nelder-Mead out-of-fold weight optimization—to enhance prediction combination, reduce multicollinearity bias, and improve the champion stacking-assisted ensembling score.

---

## 1. Architectural Components

### A. Dataset Shapes & Feature Concatenation
* Input out-of-fold prediction matrices: `Xoof` shape `(577347, 18)` (6 models × 3 target classes).
* Input test predictions: `Xtest` shape `(247435, 18)`.
* Base models: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, HistGradientBoosting.

### B. Nested 5-Fold Stratified Cross-Validation
All candidates will be trained and evaluated using the exact same outer stratified splits to prevent target label leakage:
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### C. Stacker Candidates & Settings
1. **`lr` (Baseline Stacker):**
   * Class-balanced multinomial Logistic Regression.
   * `LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced', random_state=42)`
2. **`elasticnet` (Linear Stacker):**
   * SAGA solver with L1 and L2 penalties combined.
   * `LogisticRegression(max_iter=2000, penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.1, class_weight='balanced', random_state=42)`
3. **`lgb_meta` (Tree Stacker):**
   * Shallow LightGBM tree to capture coarse interactions while preventing deep-branch overfitting on correlated signals.
   * `lgb.LGBMClassifier(max_depth=2, n_estimators=100, learning_rate=0.03, reg_alpha=5.0, reg_lambda=5.0, class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1)`
4. **`xgb_meta` (Tree Stacker):**
   * Shallow XGBoost tree matching LightGBM configurations.
   * `xgb.XGBClassifier(max_depth=2, n_estimators=100, learning_rate=0.03, reg_alpha=5.0, reg_lambda=5.0, random_state=42, n_jobs=-1, eval_metric='mlogloss')`
5. **`nm_blend` (Direct Blender):**
   * Directly searches for the optimal 6-model weights $w = [w_1, \dots, w_6]$ to maximize Balanced Accuracy on the OOF training fold, completely bypassing model ensembling layers.
   * The blended probability is:
     $$P_{\text{blend}} = \sum_{i=1}^{6} w_i \cdot P_i \quad \text{s.t.} \quad w_i \ge 0, \sum w_i = 1$$

---

## 2. Implementation details

We will create a python script `scratch/update_blender_v12.py` to programmatically rewrite `notebooks/03_model_tuning_and_ensemble.ipynb` to update Cell 9 (meta-learner evaluation) and Cell 10 (dynamic selection and test inference).

### Step-by-Step Refactoring
1. **Imports Update:** Ensure `scipy.optimize.minimize` and all classifier libraries are globally loaded.
2. **Objective Function for `nm_blend`:** Add the inner optimization loop for Nelder-Mead blending weights within the ensembling cell.
3. **Evaluation Loop:** Concatenate predictions for all candidate models and log out-of-fold Balanced Accuracy.
4. **Dynamic Selection Logic:** Pick the best scoring meta-learner, perform final Nelder-Mead threshold optimization, and execute inference on the test set.

---

## 3. Verification Plan

### Automated Verification
* Verify that the updated notebook is valid JSON:
  ```bash
  python3 -c "import json; json.load(open('notebooks/03_model_tuning_and_ensemble.ipynb'))"
  ```
* Execute the updated blender kernel locally on a 10% sample to confirm no run-time crash occurs.
