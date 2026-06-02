# Competition Instructions: Predicting Stellar Class (S6E6)

## 1. Overview
The goal of this competition is to predict the stellar class of celestial objects based on astronomical data. The targets are:
- `GALAXY`
- `STAR`
- `QSO` (Quasar)

## 2. Evaluation Metric
The evaluation metric is **Balanced Accuracy**.
$$\text{Balanced Accuracy} = \frac{\text{Recall}(\text{GALAXY}) + \text{Recall}(\text{QSO}) + \text{Recall}(\text{STAR})}{3}$$
This is the average of recall scores for each class. It corrects for class imbalance.

## 3. Data Description
The dataset contains the following files in the `data/` folder:
* `train.csv`: Training set including the labels (`class`).
* `test.csv`: Test set for which predictions need to be made.
* `sample_submission.csv`: Standard submission format.

### Features
* `id`: Unique identifier.
* `alpha`: Right ascension angle (degrees).
* `delta`: Declination angle (degrees).
* `u`: Ultraviolet filter magnitude.
* `g`: Green filter magnitude.
* `r`: Red filter magnitude.
* `i`: Near-infrared filter magnitude.
* `z`: Infrared filter magnitude.
* `redshift`: Redshift value (crucial physical signature).
* `spectral_type`: Categorical spectral classification of the object (`M`, `A/F`, `G/K`, `O/B`).
* `galaxy_population`: Categorical population of the galaxy (`Red_Sequence`, `Blue_Cloud`).
* `class`: Target class (`GALAXY`, `QSO`, `STAR`).

## 4. Key Deadlines & Constraints
* **Deadline**: June 30, 2026.
* **Internet**: Reruns/final submissions on Kaggle must be offline-safe.
