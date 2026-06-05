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

## 5. Kaggle Accounts & Submission Quota Management

To maximize the number of daily submissions (each account has 5 submissions per day), follow this protocol:

1. **First-Priority Account (`tuannm3823`)**:
   * Always prioritize running notebooks and submitting scores from this secondary account first.
   * **Important**: Make sure `tuannm3823` has accepted the rules/terms of the **`flexonafft/stellar-data`** dataset on Kaggle. Otherwise, the dataset will fail to mount to `/kaggle/input/stellar-data/external/submissions` and the ensembling notebook will fall back to local-only predictions (reducing the public score from `0.97060` to `0.96571`).
2. **Fallback Account (`tuannm3812`)**:
   * Use this primary account for running and submitting scores once `tuannm3823` reaches its daily limit.
   * This account is pre-approved for `flexonafft/stellar-data` and mounts it successfully to produce the full `0.97060` score.

### Local API Credentials Management
* Local Kaggle API credentials are automatically managed via the backup files:
  * Primary: `~/.kaggle/kaggle.tuannm3812.current.json`
  * Secondary: `~/.kaggle/kaggle.tuannm3823.backup.json`
* Use or adapt the script [push_and_submit_notebook_3823.py](file:///Users/tuanm.nguyen/Documents/kaggle-s6e6-predicting-stellar-class/scratch/push_and_submit_notebook_3823.py) to push and submit. The script handles the swap automatically and restores the primary account `tuannm3812` to `~/.kaggle/kaggle.json` in a `finally` block to prevent accidental overrides.

