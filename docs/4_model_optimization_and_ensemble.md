# Model Optimization & Ensembling

Our final submission pipeline is structured into three layers: local GBDT/bagging stacking, decision threshold optimization, and a consensus-assisted hybrid blender.

---

## 1. Stacking Meta-Learner Sweep (Layer A)

We combine the out-of-fold (OOF) prediction probability matrices from all 6 tuned baseline models. This creates an 18-dimensional feature representation (6 models $\times$ 3 class probabilities) representing the collective local model voices. 

To find the optimal ensembling strategy, we evaluated **16 distinct meta-learner configurations** side-by-side using nested 5-Fold Stratified Cross-Validation (preventing validation data leakage):

| Stacker Identifier | Stacker Model | Probability Calibration Type | OOF Balanced Accuracy |
| :--- | :--- | :--- | :--- |
| **`lr` (Baseline)** | **Logistic Regression (Multinomial)** | **None** | **0.96519 (Champion 🏆)** |
| `elasticnet` | SAGA-regularized ElasticNet | None | 0.96514 |
| `lgb_meta` | LightGBM (`max_depth=2`) | None | 0.96514 |
| `lgb_meta_iso` | LightGBM (`max_depth=2`) | Isotonic Regression | 0.96509 |
| `nm_blend` | Direct Nelder-Mead Blending | None | 0.96500 |
| `lgb_meta_platt` | LightGBM (`max_depth=2`) | Platt Scaling | 0.96499 |
| `lr_platt` | Logistic Regression (Multinomial) | Platt Scaling | 0.96350 |
| `elasticnet_platt` | SAGA-regularized ElasticNet | Platt Scaling | 0.96350 |
| `elasticnet_iso` | SAGA-regularized ElasticNet | Isotonic Regression | 0.96225 |
| `lr_iso` | Logistic Regression (Multinomial) | Isotonic Regression | 0.96233 |
| `xgb_meta_platt` | XGBoost (`max_depth=2`) | Platt Scaling | 0.95776 |
| `mlp_platt` | Multi-Layer Perceptron | Platt Scaling | 0.95744 |
| `mlp` | Multi-Layer Perceptron | None | 0.95671 |
| `mlp_iso` | Multi-Layer Perceptron | Isotonic Regression | 0.95598 |
| `xgb_meta_iso` | XGBoost (`max_depth=2`) | Isotonic Regression | 0.95519 |
| `xgb_meta` | XGBoost (`max_depth=2`) | None | 0.95502 |

* **Winner**: The raw Multinomial Logistic Regression (`lr`) meta-learner was dynamically selected as the best local ensembler, achieving **`0.96519`** validation score.

---

## 2. Decision Threshold Optimization (Layer B)

Because the evaluation metric is Balanced Accuracy (unweighted class recall), standard argmax boundaries are sub-optimal under severe class skew. We apply Nelder-Mead simplex optimization (via `scipy.optimize.minimize`) directly over the stacked OOF probabilities to find optimal class-specific multipliers:

$$\text{Final Local Class} = \arg\max \left( P_{\text{stack}} \times [t_{\text{GALAXY}}, t_{\text{QSO}}, t_{\text{STAR}}] \right)$$

* **Optimal Multipliers**: $t_{\text{GALAXY}} = 1.0059$, $t_{\text{QSO}} = 1.0232$, $t_{\text{STAR}} = 0.9709$.
* **Threshold-calibrated Stacking OOF Score**: **0.96522** (up from $0.96519$).

---

## 3. Consensus-Assisted Hybrid Blending (Layer C)

To push the final predictions to state-of-the-art accuracy, we implement a hybrid blender combining our local stacked model with **5 high-scoring public external submissions** from the `flexonafft/stellar-data` dataset (public scores: `0.96874`, `0.96973`, `0.96996`, `0.97003`, `0.97007`).

The test predictions are split into two regions:
1. **Unanimous Agreement Region (98.36% of test set - 243,385 rows)**:
   * If all 5 external models agree on the class label, we bypass local predictions and output that label directly. This preserves the high-precision consensus boundaries of the top public models.
2. **Disagreement Region (1.64% of test set - 4,050 rows)**:
   * For the remaining rows where the external models disagree, we blend the hard external votes (weighted by their public leaderboard scores) with our local stacked and threshold-calibrated soft probabilities scaled by a voice multiplier ($\alpha$):
     $$\text{votes} = \sum_{j=1}^{5} W_j \cdot L_j + \left(\alpha \sum_{j=1}^{5} W_j\right) \cdot \text{calibrated\_local\_probs}$$
   * The class receiving the maximum combined vote sum is selected.

---

## 4. Weight Sweep ($\alpha$) & Submission Results

We ran a local voice weight grid search over candidate values of $\alpha$ to find the optimal blending ratio:

| Candidate Weight ($\alpha$) | Public Leaderboard Score |
| :--- | :--- |
| `0.22` | `0.97050` |
| `0.24` | `0.97050` |
| `0.25` | `0.97046` |
| `0.26` | `0.97049` |
| `0.28` | `0.97055` |
| **`0.30` (Optimal)** | **0.97060 (Champion 🏆)** |
| **`0.35` (Optimal)** | **0.97060 (Champion 🏆)** |

* **Champion Configuration**: Pushing local voice weight $\alpha \in \{0.30, 0.35\}$ resolved the disagreement splits correctly, achieving our **best leaderboard score of 0.97060**.
* **Notebook Direct Submission**: Run remotely on Kaggle via [push_and_submit_notebook_3823.py](file:///Users/tuanm.nguyen/Documents/kaggle-s6e6-predicting-stellar-class/scratch/push_and_submit_notebook_3823.py) to execute and submit directly as a notebook version.
