# Design Specification: Insightful and Strategic Notebook Markdowns

**Date:** 2026-06-02  
**Topic:** Elevating notebook markdown narrative for stellar classification  

---

## 1. Goal Description
The objective is to refine the three Jupyter notebooks in the repository:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_baseline_modeling.ipynb`
3. `notebooks/03_model_tuning_and_ensemble.ipynb`

We will replace housekeeping notes and basic headers with detailed, insightful, and strategic markdown narrative. The notebooks will act as a publication-grade scientific and engineering report. They will cover the cosmological physics of stellar bodies, coordinate transformation geometry, ML metric calibration math, framework-specific sample-weight constraints, and leakage-proof ensembling.

---

## 2. Proposed Changes

### Component 1: `notebooks/01_eda.ipynb`
We will rewrite the markdown cells to:
- Formulate the **Balanced Accuracy** metric mathematically.
- Explain the physical cosmology of **Redshift ($z$)** and how stars, galaxies, and quasars populate different redshift bounds.
- Explain the physics of **Photometric Color Indices** as distance-invariant estimators of blackbody temperature and emission line features.
- Address the spherical geometry wrap-around issue of **Right Ascension ($\alpha$)** and **Declination ($\delta$)** coordinates at the $0^\circ/360^\circ$ boundary, showing the trigonometric unit-sphere projection formulas.

### Component 2: `notebooks/02_baseline_modeling.ipynb`
We will rewrite the markdown cells to:
- Elaborate on the **Stratified K-Fold validation strategy** and explain how it prevents validation bias when faced with severe target class skew.
- Detail the **Feature Engineering Pipeline** transformations.
- Formulate the framework-specific **Sample Weighting** and **Class Weighting** strategies for LightGBM, CatBoost, and XGBoost to directly target Balanced Accuracy.
- Summarize the diverse tree-splitting architectures (Leaf-wise vs Level-wise vs Symmetric trees) and their ensembling compatibility.

### Component 3: `notebooks/03_model_tuning_and_ensemble.ipynb`
We will rewrite the markdown cells to:
- Detail the mathematics of **Probability Blending** (soft voting) for variance reduction.
- Explain **Data Leakage Protection** by running grid search weight optimization strictly on Out-of-Fold (OOF) validation predictions.
- Outline the post-processing inverse mapping and directory/environment configuration setup.

---

## 3. Verification Plan
- **Syntax Validation:** Verify that all notebooks are valid JSON and can be parsed by `json.load()` without errors. Note that backslashes in LaTeX equations (like `\alpha`, `\delta`, `\approx`, `\frac`) must be double-escaped (e.g., `\\alpha`, `\\delta`, `\\approx`, `\\frac`) inside the JSON notebook structure.
- **Portability:** Verify that the dynamic path resolution triggers correctly for local vs Kaggle paths.
