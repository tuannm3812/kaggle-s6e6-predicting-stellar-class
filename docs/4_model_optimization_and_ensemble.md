# Model Optimization & Ensembling

## 1. Blending Strategy
To maximize the final Balanced Accuracy, we combine the out-of-fold (OOF) prediction probabilities from the three baseline models using a weighted average:

$$P_{\text{blend}} = w_{\text{LGB}} P_{\text{LGB}} + w_{\text{XGB}} P_{\text{XGB}} + w_{\text{Cat}} P_{\text{Cat}}$$

By performing a grid search strictly on the OOF predictions (a leakage-free validation set), we prevent overfitting to the training distribution.

## 2. Ensemble Weights Optimization
The grid search (step size 0.05, subject to $\sum w = 1.0$) identified the following optimal blending configuration:

* **LightGBM Weight**: $0.50$
* **XGBoost Weight**: $0.40$
* **CatBoost Weight**: $0.10$

**Ensemble OOF Balanced Accuracy**: **0.96566**

### Insights
* Blending the models yielded a $+0.00030$ absolute improvement over the single best model (XGBoost at $0.96536$).
* LightGBM received the highest weight despite XGBoost having a marginally higher individual CV score, highlighting that LightGBM's leaf-wise splits provided complementary orthogonal predictions to XGBoost's depth-wise splits.

## 3. Decision Threshold Calibration
Because the metric is Balanced Accuracy, standard argmax boundaries are sub-optimal for skewed classes. We utilized Nelder-Mead Simplex optimization to discover class-specific probability multipliers before taking the argmax.

## 4. Final Inference
The optimal blending weights and decision threshold multipliers were applied to the combined Test Set predictions. The finalized predicted labels were mapped back to their text values (`GALAXY`, `QSO`, `STAR`) and saved successfully to `submission.csv`.
