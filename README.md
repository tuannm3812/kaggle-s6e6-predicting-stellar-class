<div align="center">
  <img src="https://classic.sdss.org/segue/Segue-4135-6-58_60.png" width="100%" alt="SDSS Segue Stellar Banner" />
  <h1>Stellar Classification: Predicting Astronomical Objects</h1>
  <p><b>Kaggle Playground Series - Season 6, Episode 6</b></p>
</div>

---

## 🌌 Project Overview
This repository implements an end-to-end machine learning pipeline to classify celestial objects into three target classes—**`GALAXY`**, **`STAR`**, and **`QSO` (Quasar)**—using photometric and positional data derived from SDSS (Sloan Digital Sky Survey)-like observations. 

Our pipeline is structured to directly optimize **Balanced Accuracy** (the unweighted average of class recalls) and handles synthetic data artifacts, coordinate wrapping, and target class skew directly within a cross-validated tree-based framework.

---

## 🛠️ Repository Structure
In alignment with workflow portability, the repository uses a **notebook-centric** architecture. All modeling and evaluation logic is self-contained in portable Jupyter notebooks designed to execute seamlessly on Kaggle and locally:

```text
.
├── README.md                  # Project overview and pipeline documentation
├── requirements.txt           # Python dependencies
├── .gitignore                 # Configured to exclude prediction dumps and venv/
├── docs/                      # Strategic writeups & coding guidelines
│   ├── coding_standards.md    # Code styling and formatting rules (PEP 8, Viridis)
│   ├── 1_instructions.md      # Rules and metric evaluations
│   └── superpowers/           # Design specifications and implementation plans
└── notebooks/                 # Self-contained executable notebooks
    ├── 01_eda.ipynb           # Deep-dive exploratory data analysis
    ├── 02_baseline_modeling.ipynb  # 5-fold Stratified CV baseline training
    └── 03_model_tuning_and_ensemble.ipynb # Weight optimization & submission generation
```

---

## 📊 Core EDA Insights & Modeling Strategy
Our exploratory analysis uncovered critical physical and synthetic properties that directly dictate our modeling choices:

1. **Balanced Accuracy Metric Alignment**
   * *Insight:* The dataset is highly imbalanced (`GALAXY`: ~65.4%, `QSO`: ~20.3%, `STAR`: ~14.3%). In a Balanced Accuracy metric, a misclassification on a `STAR` is **~4.5x** more costly than on a `GALAXY`.
   * *Strategy:* We configure inverse target weights (`class_weight='balanced'` in LightGBM, `auto_class_weights='Balanced'` in CatBoost, and custom fold-level sample weights in XGBoost).
2. **Cosmological Redshift ($z$) Artifacts**
   * *Insight:* bound Milky Way stars physically have redshifts very close to zero. However, in this CTGAN-generated dataset, **75% of stars have redshifts > 0.02**, stretching up to **5.44**.
   * *Strategy:* Standard astronomical physics cutoffs (e.g. $z \le 0.001$) fail. The tree models must learn to resolve high-redshift stars using multi-dimensional colors and spectral codes.
3. **Coordinate Discontinuity Resolution**
   * *Insight:* Right Ascension ($\alpha$) is a circular longitude ($0^\circ \to 360^\circ$). Tree algorithms split orthogonally, separating adjacent coordinates (like $359.9^\circ$ and $0.1^\circ$) to opposite extremes.
   * *Strategy:* We transform celestial coordinates ($\alpha, \delta$) into 3D Cartesian coordinates ($x, y, z$) on a unit sphere.
4. **Photometric Color Indices**
   * *Insight:* Raw magnitudes ($u, g, r, i, z$) depend on object distance. By engineering color indices (e.g. $u-g, g-r$), we capture distance-invariant surface temperatures and emission lines.

---

## 📈 Baseline CV Results
Our 5-fold Stratified Cross-Validation on the training set yields the following baseline scores (**Balanced Accuracy**):

| Model Framework | Tree Split Architecture | 5-Fold CV Score |
| :--- | :--- | :--- |
| **XGBoost** | Level-wise (Depth-first) | **`0.96536`** (Champion) |
| **LightGBM** | Leaf-wise (Best-first) | **`0.96512`** |
| **CatBoost** | Symmetric Oblivious Trees | **`0.96211`** |

---

## 🚀 Execution Guide on Kaggle

### Step 1: Execute Baseline Model
* Upload and execute `02_baseline_modeling.ipynb` on Kaggle. It will output validation and test probability matrices (`lgb_oof.npy`, `xgb_test.npy`, etc.) to `./predictions`.

### Step 2: Run Ensemble Optimization
* Attach Notebook 2's outputs as a dataset to Notebook 3 (`03_model_tuning_and_ensemble.ipynb`).
* Notebook 3 is configured to directly read the prediction files from:
  `/kaggle/input/notebooks/tuannm3812/stellar-classification-baseline-modeling/predictions`
* Run Notebook 3 to optimize the ensembling blend weights ($w_{\text{LGB}}, w_{\text{XGB}}, w_{\text{Cat}}$) on OOF predictions (preventing data leakage) and generate the final `submission.csv`.
