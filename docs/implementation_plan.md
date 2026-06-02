# Implementation Plan: Predicting Stellar Class (Kaggle S6E6)

This document outlines the end-to-end plan to solve the Predicting Stellar Class competition. The objective is to predict the `class` of astronomical objects (`GALAXY`, `STAR`, or `QSO`) based on photometrical and positional data. The competition uses **Balanced Accuracy** as the evaluation metric.

In alignment with your project standards, this workflow will follow a notebook-first exploration paired with structured documentation in `docs/`, clean coding standards, and visual consistency using the `Viridis` palette.

---

## 1. Project Overview & Data Context

From the initial inspection, the dataset has the following structure:
* **Train Set**: 577,347 samples, 12 columns.
* **Test Set**: 247,435 samples, 11 columns (excluding `class`).
* **Missing Values**: None in either train or test sets.
* **Target Distribution**:
  * `GALAXY`: 377,480 (~65.4%)
  * `QSO`: 117,143 (~20.3%)
  * `STAR`: 82,724 (~14.3%)
  * *Note: The dataset is significantly imbalanced, which makes optimizing for Balanced Accuracy critical.*

### Expected Kaggle input path:
```text
/kaggle/input/competitions/playground-series-s6e6
```

### Input files:
* `train.csv`: Training set including the labels (`class`).
* `test.csv`: Test set for which predictions need to be made.
* `sample_submission.csv`: Standard submission format.

---

## 2. Repository Structure

We will organize the repository as follows:
```text
.
├── README.md                  # Project overview and current champion results
├── requirements.txt           # Python dependencies
├── .gitignore                 # Excluded directories (data/, predictions/, venv/)
├── data/                      # Local data (train.csv, test.csv, sample_submission.csv)
├── docs/                      # Markdown documentation of findings and decisions
│   ├── coding_standards.md    # Repository coding guidelines
│   ├── 1_instructions.md      # Competition rules and instructions
│   ├── 2_eda_insights.md      # Data quality, target behavior, and features analysis
│   ├── 3_baseline_modeling.md # CV setup and comparison of baseline models
│   ├── 4_model_optimization_and_ensemble.md  # Hyperparameter tuning, feature validation, and ensembling
│   └── implementation_plan.md # Copy of the implementation plan
└── notebooks/                 # Jupyter notebooks for interactive development
    ├── 01_eda.ipynb           # Exploratory data analysis
    ├── 02_baseline_modeling.ipynb  # Baseline model comparison
    └── 03_model_tuning_and_ensemble.ipynb # Optimization, feature validation, and blending
```

---

## 3. Workflow & Notebook Sequence

| Phase / Notebook | Purpose & Output |
| --- | --- |
| **01_eda.ipynb** | Analyze feature distributions, target class behavior, coordinates mapping, color features, and potential drift. Output: [docs/2_eda_insights.md](file:///Users/tuanm.nguyen/Documents/kaggle-s6e6-predicting-stellar-class/docs/2_eda_insights.md). |
| **02_baseline_modeling.ipynb** | Setup stratified 5-fold CV. Train baseline LightGBM, XGBoost, and CatBoost models. Output: [docs/3_baseline_modeling.md](file:///Users/tuanm.nguyen/Documents/kaggle-s6e6-predicting-stellar-class/docs/3_baseline_modeling.md). |
| **03_model_tuning_and_ensemble.ipynb** | Tune hyperparameters, validate engineered features (color indexes, coordinate transforms), blend predictions, calibrate probabilities, and write `submission.csv`. Output: [docs/4_model_optimization_and_ensemble.md](file:///Users/tuanm.nguyen/Documents/kaggle-s6e6-predicting-stellar-class/docs/4_model_optimization_and_ensemble.md). |

---

## 4. Coding & Design Standards

We will strictly maintain the following guidelines:
1. **PEP 8 Compliance**: Use 4 spaces for indentation, blank lines to separate import groups, and keep lines to 79 characters where practical.
2. **Google-style Docstrings & Type Hints**: All reusable python functions will have explicit type signatures and detailed docstrings explaining their arguments and returns.
3. **Viridis Colormap**: All plots in notebooks will default to the `"viridis"` color palette or colormap for consistency and readability.
4. **Clean Notebooks**: Run notebooks end-to-end, clear unnecessary output sizes before committing, and make cells self-contained.

---

## 5. Feature Engineering Themes

We will explore and validate the following features:
* **Color Indices**: Successive and non-successive magnitude differences:
  * `u - g`, `g - r`, `r - i`, `i - z`
  * `u - r`, `g - i`, `r - z`, `u - i`, `g - z`, `u - z`
* **3D Celestial Coordinates**:
  * Convert RA (`alpha`) and Dec (`delta`) from degrees to radians, then compute Cartesian coordinates $x, y, z$.
* **Categorical Mappings**:
  * Map `spectral_type` and `galaxy_population` using consistent mappings or native category columns.
* **Leakage Prevention**:
  * Do not use the target class `class` in feature construction. All target encoding or race/calibration steps must be computed safely inside each validation fold.

---

## 6. Validation Strategy

* **Cross-Validation**: 5-Fold Stratified Cross-Validation (maintains class ratios of 65.4% GALAXY, 20.3% QSO, 14.3% STAR).
* **Metric**: Balanced Accuracy.
* **Class Weighting**: Pass balanced weights or calculate sample weights for XGBoost to ensure the model optimizes for class-balanced recall rather than absolute majority class performance.

---

## 7. Next Steps & Action Plan
1. Create the `docs/` folder and initialize `docs/coding_standards.md` and `docs/1_instructions.md`.
2. Flesh out the EDA in `notebooks/01_eda.ipynb` using `seaborn` / `matplotlib` with the Viridis palette, and document the insights in `docs/2_eda_insights.md`.
3. Build the baseline models in `notebooks/02_baseline_modeling.ipynb`, capturing the results in `docs/3_baseline_modeling.md`.
4. Tune parameters and perform ensembling in `notebooks/03_model_tuning_and_ensemble.ipynb`, producing the final `submission.csv` and documenting it in `docs/4_model_optimization_and_ensemble.md`.
