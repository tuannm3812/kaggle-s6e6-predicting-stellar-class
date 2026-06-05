# Baseline Modeling Results

## 1. Experimental Setup
* **Validation Strategy**: 5-Fold Stratified Cross-Validation (maintaining target ratios: `GALAXY` ~65.4%, `QSO` ~20.3%, `STAR` ~14.3%).
* **Metric**: Balanced Accuracy.
* **Feature Engineering Pipeline**:
  * **J2000 Galactic Coordinate Rotation**: Projects equatorial RA/Dec ($\alpha, \delta$) to Galactic coordinates ($l, b$) using NGP reference constants.
  * **3D Celestial Cartesian Coordinates**: Maps wrapping coordinate longitude ($\alpha$) using unit sphere $(x, y, z)$ coordinates.
  * **Stellar Color Bins & Curvatures**: Calculates 10 raw color indexes (e.g. $u-g, g-r$) and 6 second-order difference-of-differences curves representing physical surface temperatures.
  * **Redshift Interaction Colors**: Constructs multiplicative interaction terms (e.g. `u_g * redshift`) to segment synthetic CTGAN anomalies.

## 2. Model Suite & Target Imbalance Handling
To ensure the models optimize for the average of class recalls (Balanced Accuracy) rather than absolute majority class performance under severe target skew, we train 6 distinct tree-based architectures with balanced class/sample weighting:
1. **LightGBM** (`lgb`): Leaf-wise (best-first) GBDT. Configured with `class_weight='balanced'`.
2. **XGBoost** (`xgb`): Level-wise (depth-first) GBDT. Trained using fold-level sample weights generated via `compute_sample_weight('balanced', y)`.
3. **CatBoost** (`cat`): Symmetric oblivious trees. Configured with `auto_class_weights='Balanced'`.
4. **Random Forest** (`rf`): Bagging classifier. Trained with `class_weight='balanced'` and deep trees (`max_depth=12`).
5. **Extra Trees** (`et`): Extremely randomized trees. Trained with `class_weight='balanced'` and deep trees (`max_depth=12`).
6. **HistGradientBoosting** (`hist_gb`): Binning-based GBDT. Configured with `class_weight='balanced'`.

All models were tuned via automated GPU-accelerated Optuna hyperparameter search loops running over stratified subsets.

## 3. Stratified 5-Fold CV Scores (Optuna-Tuned)
The final validation scores across the 5-fold CV show high stability and robust generalization:

| Model Identifier | Estimator Type | Tree Split Architecture | 5-Fold CV Balanced Accuracy |
| :--- | :--- | :--- | :--- |
| **LightGBM** | GBDT | Leaf-wise (Best-first) | **0.96490** |
| **XGBoost** | GBDT | Level-wise (Depth-first) | **0.96487** |
| **HistGradientBoosting** | GBDT | Binning Leaf-wise | **0.96459** |
| **CatBoost** | GBDT | Symmetric Oblivious | **0.96329** |
| **Random Forest** | Bagging Ensemble | Orthogonal Splits | **0.95699** |
| **Extra Trees** | Bagging Ensemble | Random Splits | **0.94616** |

### Key Observations
* The tuned GBDT models (LightGBM, XGBoost, HistGB) maintain highly aligned performance, indicating they have converged to the physical limits of the engineered color and coordinate space.
* Bagging models (Random Forest and Extra Trees) score slightly lower individually, but they introduce unique split diversity (randomized thresholds and orthogonal decision boundaries) which is highly valuable for the stacking layer.
* All out-of-fold (OOF) prediction probability files (`*_oof.npy`) and test predictions (`*_test.npy`) have been successfully saved to the predictions directory to be loaded by the stacking meta-learner.
