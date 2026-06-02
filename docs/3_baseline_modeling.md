# Baseline Modeling Results

## 1. Experimental Setup
* **Validation Strategy**: 5-Fold Stratified Cross-Validation
* **Metric**: Balanced Accuracy
* **Features Included**: Raw magnitudes, engineered color indices, 3D cartesian coordinates (x, y, z), and mapped categoricals (spectral type, galaxy population).
* **Target Mapping**: `GALAXY: 0`, `QSO: 1`, `STAR: 2`.

## 2. Model Architectures & Handling Imbalance
Because the target classes are heavily imbalanced (GALAXY: ~65%, QSO: ~20%, STAR: ~14%) and the competition evaluates on **Balanced Accuracy**, each framework was configured to apply inverse class/sample weighting during gradient updates:
* **LightGBM**: Configured with `class_weight='balanced'`.
* **XGBoost**: Trained using fold-level sample weights generated via `compute_sample_weight('balanced', y)`.
* **CatBoost**: Managed through `auto_class_weights='Balanced'`.

## 3. Results Overview
The baseline scores across the 5-fold CV are robust, showing tight generalization with minimal variance across folds:

| Model Framework | Growth Strategy | 5-Fold Balanced Accuracy |
| :--- | :--- | :--- |
| **XGBoost** | Level-wise (Depth-first) | **0.96536** |
| **LightGBM** | Leaf-wise (Best-first) | **0.96512** |
| **CatBoost** | Symmetric Oblivious Trees | **0.96211** |

### Insights
* XGBoost marginally outperforms LightGBM.
* All models successfully handled the target imbalance thanks to class-weight adjustments.
* Out-of-Fold (OOF) and Test probability predictions have been exported for ensembling.
