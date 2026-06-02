# Exploratory Data Analysis (EDA) Insights

## 1. Data Quality and Completeness
* **Train Set**: 577,347 rows, 12 columns.
* **Test Set**: 247,435 rows, 11 columns (excluding `class`).
* **Missing Values**: 0 missing values across all columns in both train and test splits.
* **Duplication**: No duplicate entries or ID overlaps were found.

## 2. Target Class Distribution
The target variable `class` contains three distinct categories with significant class imbalance:

| Class | Count | Percentage |
| --- | --- | --- |
| `GALAXY` | 377,480 | 65.38% |
| `QSO` (Quasar) | 117,143 | 20.29% |
| `STAR` | 82,724 | 14.33% |

*Implication*: Because the evaluation metric is **Balanced Accuracy**, models trained on raw probabilities will over-predict `GALAXY`. Class weights, sample weights, or post-tuning threshold calibration are required to ensure the model focuses equally on recall for `QSO` and `STAR`.

## 3. High-Signal Features

### A. Redshift
* `redshift` is the single most predictive feature. It represents how much the light wavelengths are stretched due to cosmic expansion.
  * **STARS**: Clustered closely around $0$ (median: $0.00007$, std: $0.0004$), which is expected since stars in our galaxy are not subject to Hubble expansion.
  * **GALAXIES**: Distributed in the low-to-mid range (median: $0.077$, std: $0.043$).
  * **QSOs (Quasars)**: Distant, high-energy active galactic nuclei. They have significantly higher redshift values (median: $1.72$, std: $0.78$), extending up to $7+$.
* *Implication*: Tree models will find clean, early splits on `redshift` to isolate `STAR` and `QSO`.

### B. Astronomical Colors
* Magnitudes across filters `u`, `g`, `r`, `i`, `z` represent the light intensity. However, their raw values depend heavily on the distance and luminosity.
* By computing color indexes (e.g. `u - g`, `g - r`, `r - i`, `i - z`), we remove distance effects and isolate the physical temperature/spectral shape of the objects.
* Plotting color-color diagrams (such as `u - g` vs `g - r`) reveals distinct, separable cluster bands for Stars, Quasars, and Galaxies.

### C. Celestial Coordinates
* `alpha` (Right Ascension) and `delta` (Declination) represent the positions on the sky sphere.
* Direct mapping shows standard coordinate coverage. Since coordinates wrap ($360^\circ$ RA), we convert these to 3D Cartesian coordinates ($x, y, z$) to prevent boundary issues at $0^\circ / 360^\circ$ RA.

### D. Categorical Associations
* **Spectral Type**:
  * `M` spectral class represents colder stars/galaxies and is the most common. It is heavily associated with the `GALAXY` target.
  * `O/B` represents very hot, blue stars or quasars, heavily associated with the `QSO` target.
* **Galaxy Population**:
  * `Red_Sequence` contains older, redder galaxies (mostly GALAXY class).
  * `Blue_Cloud` contains star-forming, bluer galaxies (more QSO/GALAXY class).
