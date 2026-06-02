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

---

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

---

## 4. Deep-Dive Scientific Analysis

### I. Redshift Boundary overlaps
Although `redshift` is a dominant feature, there are specific physically motivated zones where class overlap occurs:
1. **Low-Redshift Galaxies vs. Stars ($z \approx 0$)**: Local Group galaxies (like Andromeda or satellite dwarfs) have redshifts close to 0, or even slightly negative due to local gravitational attraction towards our Milky Way. A model relying purely on $z = 0$ will misclassify these local galaxies as Stars. Magnitudes and colors are needed to break this tie (galaxies are spatially extended and have different color curves than individual stars).
2. **High-Redshift Galaxies vs. Quasars ($z \ge 1.0$)**: Very distant, active star-forming galaxies can show high redshift values overlapping with the Quasar distribution. Here, the categorical indicators (`spectral_type = O/B` and `galaxy_population = Blue_Cloud` for Quasars) play a deciding role.

### II. The Physics of Color Indices
The SDSS filter bands (`u`, `g`, `r`, `i`, `z`) capture light across different spectral ranges:
* `u` (Ultraviolet, $\sim 354$ nm)
* `g` (Green, $\sim 475$ nm)
* `r` (Red, $\sim 622$ nm)
* `i` (Near-Infrared, $\sim 763$ nm)
* `z` (Infrared, $\sim 905$ nm)

Color differences act as temperature indicators:
* **Stellar Blackbody curves**: Cooler stars (Class M) peak in the red/infrared and have large positive values for `u - g` and `g - r`. Hotter stars (Class O/B) peak in the UV and have negative/very small values for `u - g`.
* **Quasar Emission Lines**: Quasars deviate from simple stellar blackbody curves because they have flat power-law spectra with strong emission lines (like Hydrogen Lyman-alpha). This puts them in a unique region of color-color space, especially when plotting `u - g` versus `g - r`.

### III. Coordinate Wrapping Geometry
Celestial coordinates are angular coordinates mapped to a sphere:
* Right Ascension (`alpha` or $\alpha$) is the celestial equivalent of longitude, ranging from $0^\circ$ to $360^\circ$.
* Declination (`delta` or $\delta$) is the celestial equivalent of latitude, ranging from $-90^\circ$ to $+90^\circ$.

Because $\alpha$ wraps around (i.e., $359.9^\circ$ is physically adjacent to $0.1^\circ$), standard tree-based models (LightGBM, XGBoost, CatBoost) struggle with this discontinuity. A split at $\alpha = 180^\circ$ splits the circle, treating the boundary region as opposite extremes.

By transforming the coordinates into 3D Cartesian coordinates ($x, y, z$) on a unit sphere, we resolve the discontinuity:
$$x = \cos(\delta) \times \cos(\alpha)$$
$$y = \cos(\delta) \times \sin(\alpha)$$
$$z = \sin(\delta)$$
*(Angles are converted to radians first)*. This maps close coordinates to close points in $x, y, z$ space, allowing tree splits to isolate spatial regions of the sky (e.g., specific survey bands or galactic plane alignments) without boundary errors.
