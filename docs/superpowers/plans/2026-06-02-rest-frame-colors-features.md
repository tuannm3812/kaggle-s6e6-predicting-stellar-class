# Rest-Frame Color Index Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 10 rest-frame color index features corrected for cosmological redshift wavelength stretching in Notebook 2's feature engineering pipeline to improve model predictive power.

**Architecture:** Use a python script to programmatically load and modify `notebooks/02_baseline_modeling.ipynb` and `notebooks/01_eda.ipynb` (so EDA remains aligned). We will replace the source of the `feature_engineering()` function to append the rest-frame features.

**Tech Stack:** Python 3, JSON, Pandas, NumPy

---

## Proposed Changes

### [Jupyter Notebooks]

#### [MODIFY] [02_baseline_modeling.ipynb](file:///Users/tuanm.nguyen/Documents/kaggle-s6e6-predicting-stellar-class/notebooks/02_baseline_modeling.ipynb)
- Update `feature_engineering()` function (cell index 5) to engineer `u_g_rest` through `u_z_rest` color index features scaled by $1 + \text{redshift}$.

#### [MODIFY] [01_eda.ipynb](file:///Users/tuanm.nguyen/Documents/kaggle-s6e6-predicting-stellar-class/notebooks/01_eda.ipynb)
- Update `feature_engineering()` inside any utility blocks if present (Notebook 1 doesn't have a formal feature engineering pipeline, but we will add notes to its takeaways cell explaining rest-frame color indices).

---

### Task 1: Create and Run Baseline Update Script

**Files:**
- Create: `scratch/update_baseline_features.py`

- [ ] **Step 1: Write python update script**
  Create a script `scratch/update_baseline_features.py` to modify Notebook 2.

  ```python
  import json

  def update_baseline_notebook():
      path = "notebooks/02_baseline_modeling.ipynb"
      print(f"Loading {path}...")
      with open(path, "r", encoding="utf-8") as f:
          data = json.load(f)
      
      # We modify cell index 5
      cell = data["cells"][5]
      assert cell["cell_type"] == "code"
      
      cell["source"] = [
          "def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:\\n",
          "    \\\"\\\"\\\"Applies feature engineering to astronomical features.\\n",
          "\\n",
          "    Args:\\n",
          "        df: Input raw DataFrame.\\n",
          "\\n",
          "    Returns:\\n",
          "        DataFrame with mapped categoricals, colors, and 3D coordinates.\\n",
          "    \\\"\\\"\\\"\\n",
          "    df = df.copy()\\n",
          "    \\n",
          "    # Categoricals mapping\\n",
          "    df['spectral_type_code'] = df['spectral_type'].map(SPECTRAL_MAP)\\n",
          "    df['galaxy_pop_code'] = df['galaxy_population'].map(GALAXY_POP_MAP)\\n",
          "    \\n",
          "    # Astronomical Colors (differences in magnitudes)\\n",
          "    # Successive filters\\n",
          "    df['u_g'] = df['u'] - df['g']\\n",
          "    df['g_r'] = df['g'] - df['r']\\n",
          "    df['r_i'] = df['r'] - df['i']\\n",
          "    df['i_z'] = df['i'] - df['z']\\n",
          "    \\n",
          "    # Non-successive filters\\n",
          "    df['u_r'] = df['u'] - df['r']\\n",
          "    df['g_i'] = df['g'] - df['i']\\n",
          "    df['r_z'] = df['r'] - df['z']\\n",
          "    df['u_i'] = df['u'] - df['i']\\n",
          "    df['g_z'] = df['g'] - df['z']\\n",
          "    df['u_z'] = df['u'] - df['z']\\n",
          "    \\n",
          "    # Rest-frame color indices (corrected for cosmological redshift wavelength stretching)\\n",
          "    df['u_g_rest'] = df['u_g'] / (1 + df['redshift'])\\n",
          "    df['g_r_rest'] = df['g_r'] / (1 + df['redshift'])\\n",
          "    df['r_i_rest'] = df['r_i'] / (1 + df['redshift'])\\n",
          "    df['i_z_rest'] = df['i_z'] / (1 + df['redshift'])\\n",
          "    df['u_r_rest'] = df['u_r'] / (1 + df['redshift'])\\n",
          "    df['g_i_rest'] = df['g_i'] / (1 + df['redshift'])\\n",
          "    df['r_z_rest'] = df['r_z'] / (1 + df['redshift'])\\n",
          "    df['u_i_rest'] = df['u_i'] / (1 + df['redshift'])\\n",
          "    df['g_z_rest'] = df['g_z'] / (1 + df['redshift'])\\n",
          "    df['u_z_rest'] = df['u_z'] / (1 + df['redshift'])\\n",
          "    \\n",
          "    # Trigonometric transformation of celestial coordinates (Right Ascension & Declination)\\n",
          "    alpha_rad = np.radians(df['alpha'])\\n",
          "    delta_rad = np.radians(df['delta'])\\n",
          "    df['coord_x'] = np.cos(delta_rad) * np.cos(alpha_rad)\\n",
          "    df['coord_y'] = np.cos(delta_rad) * np.sin(alpha_rad)\\n",
          "    df['coord_z'] = np.sin(delta_rad)\\n",
          "    \\n",
          "    # Drop raw IDs and raw categorical columns\\n",
          "    cols_to_drop = ['id', 'spectral_type', 'galaxy_population']\\n",
          "    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])\\n",
          "    \\n",
          "    return df"
      ]
      
      with open(path, "w", encoding="utf-8") as f:
          json.dump(data, f, indent=1)
      print(f"Successfully updated {path}!")

  if __name__ == "__main__":
      update_baseline_notebook()
  ```

- [ ] **Step 2: Run update script**
  Run: `python3 scratch/update_baseline_features.py`

- [ ] **Step 3: Validate Notebook 2 JSON syntax**
  Run: `python3 -c "import json; json.load(open('notebooks/02_baseline_modeling.ipynb')); print('Notebook 2 is valid!')"`
