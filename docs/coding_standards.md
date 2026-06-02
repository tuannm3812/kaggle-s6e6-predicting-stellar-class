# Coding Standards

## 1. Repository Scope

This repository balances both local Python module training and Jupyter notebooks for interactive development:

- `notebooks/` for exploratory analysis and interactive experiments.
- `docs/` for standards, instructions, EDA notes, modeling results, and ensemble summaries.
- `src/` for clean, reusable production and training logic (e.g. `train.py`).
- `data/` for local CSV data (git ignored, not to be committed).
- `README.md` for high-level overview and current champion results.

## 2. Notebook Naming

Use numbered, stable notebook names describing the workflow phase:

1. `01_eda.ipynb`
2. `02_baseline_modeling.ipynb`
3. `03_model_tuning_and_ensemble.ipynb`

## 3. Code Style

Follow PEP 8 for Python code:

- Use 4 spaces for indentation.
- Keep lines to 79 characters or fewer where practical.
- Prefer f-strings, list comprehensions, and small utility functions.
- Add type hints for all reusable functions.
- Group imports in this order:
  1. Standard library
  2. Third-party libraries
  3. Local modules (e.g., from `src`)
- Separate import groups with a blank line.

Use Google-style docstrings for reusable functions:

```python
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies feature engineering to astronomical features.

    Args:
        df: Input raw DataFrame.

    Returns:
        DataFrame with mapped categoricals, colors, and 3D coordinates.
    """
```

## 4. Notebook Style

Each notebook should include:

- a short purpose statement at the top;
- a configuration section near the top for tunable parameters (e.g. `N_SPLITS`, `SEED`, model configs);
- Markdown insight cells after major plots or metrics;
- Output cell clearing before committing to keep notebooks lightweight.

## 5. Feature Engineering Standards

- Features must be constructed only from attributes available in both train and test sets.
- Avoid target leakage: do not use the target `class` in any global feature transformations. Target encoding or other target-based metrics must be fit and applied strictly within each cross-validation fold.

## 6. Plot Style

Use the **Viridis** palette as the default visual language across notebooks:

- Use `sns.color_palette("viridis", ...)` for categorical or sequential color schemes.
- Use `"viridis"` as the default colormap for heatmaps and scatter plots.
- Keep titles and labels concise and analytical.

## 7. Documentation Style

- Lead with findings and implications.
- Include exact metrics (Balanced Accuracy).
- Link notebooks and documentation using relative links.

## 8. Git Hygiene

Do not commit:
- Raw Kaggle data (`data/*.csv` or `data/*.zip`).
- Large checkpoints, model binary dumps, or prediction numpy arrays (`predictions/`).
- Jupyter checkpoints or local virtual environments (`venv/`).
