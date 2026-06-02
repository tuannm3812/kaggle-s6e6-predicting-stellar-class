import json
import os

def update_tuning_notebook():
    path = "notebooks/03_model_tuning_and_ensemble.ipynb"
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    new_cells = []
    for i, cell in enumerate(data["cells"]):
        new_cells.append(cell)
        if i == 9: # After grid search code cell
            # 1. Add Markdown Cell
            new_cells.append({
                "cell_type": "markdown",
                "id": "threshold_tuning_md",
                "metadata": {},
                "source": [
                    "## 4.5. Decision Threshold Optimization for Balanced Accuracy\n",
                    "\n",
                    "**Threshold Weighting Theory:**\n",
                    "Standard inference assigns classes using $\\arg\\max(P_c)$. However, when optimizing for Balanced Accuracy (unweighted average recall) under severe class skew, the standard probability boundaries are mathematically sub-optimal. We can find class-specific multipliers $(t_{\\text{GALAXY}}, t_{\\text{QSO}}, t_{\\text{STAR}})$ to scale the predicted probabilities prior to argmax: \n",
                    "$$\\text{Prediction} = \\arg\\max_{c \\in \\{0, 1, 2\\}} \\left( t_c \\times P_c \\right)$\n",
                    "\n",
                    "**Nelder-Mead Simplex Optimization:**\n",
                    "Since the argmax operation is step-wise and non-differentiable, gradient descent cannot be used. Instead, we use the **Nelder-Mead simplex algorithm** (via `scipy.optimize.minimize`) to search for the multipliers that directly maximize Balanced Accuracy on our Out-of-Fold (OOF) predictions."
                ]
            })
            # 2. Add Code Cell
            new_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "id": "threshold_tuning_code",
                "metadata": {},
                "outputs": [],
                "source": [
                    "from scipy.optimize import minimize\n",
                    "\n",
                    "def optimize_thresholds(oof_probs, y_true):\n",
                    "    def loss_func(weights):\n",
                    "        # We maximize Balanced Accuracy, so we minimize its negative\n",
                    "        scaled_probs = oof_probs * weights\n",
                    "        preds = np.argmax(scaled_probs, axis=1)\n",
                    "        return -balanced_accuracy_score(y_true, preds)\n",
                    "    \n",
                    "    # Start with equal weights\n",
                    "    init_weights = [1.0, 1.0, 1.0]\n",
                    "    \n",
                    "    # Run Nelder-Mead optimization\n",
                    "    res = minimize(loss_func, init_weights, method='Nelder-Mead', options={'maxiter': 500})\n",
                    "    \n",
                    "    best_weights = res.x\n",
                    "    # Normalize weights so they sum to 3.0\n",
                    "    best_weights = best_weights / np.sum(best_weights) * 3.0\n",
                    "    best_score = -res.fun\n",
                    "    return best_weights, best_score\n",
                    "\n",
                    "# Calculate the blended OOF probabilities using optimal weights\n",
                    "blend_oof = w_lgb * lgb_oof + w_xgb * xgb_oof + w_cat * cat_oof\n",
                    "\n",
                    "print(\"--- Optimizing Decision Thresholds ---\")\n",
                    "best_thresholds, optimized_score = optimize_thresholds(blend_oof, y)\n",
                    "t_gal, t_qso, t_star = best_thresholds\n",
                    "\n",
                    "print(f\"Original Blend Score: {best_score:.5f}\")\n",
                    "print(f\"Optimized Score:      {optimized_score:.5f}\")\n",
                    "print(f\"Optimal Multipliers -> GALAXY: {t_gal:.4f}, QSO: {t_qso:.4f}, STAR: {t_star:.4f}\")"
                ]
            })
    
    final_inference_cell = new_cells[13]
    assert final_inference_cell["cell_type"] == "code"
    
    final_inference_cell["source"] = [
        "# Applying optimized blending weights to the test predictions\n",
        "final_test_prob = w_lgb * lgb_test + w_xgb * xgb_test + w_cat * cat_test\n",
        "\n",
        "# Applying optimized threshold multipliers before taking argmax\n",
        "scaled_test_prob = final_test_prob * best_thresholds\n",
        "final_preds = np.argmax(scaled_test_prob, axis=1)\n",
        "\n",
        "# Create submission DataFrame\n",
        "sub_sample_path = os.path.join(DATA_DIR, 'sample_submission.csv')\n",
        "submission = pd.read_csv(sub_sample_path)\n",
        "submission['class'] = [INV_TARGET_MAP[p] for p in final_preds]\n",
        "\n",
        "# Save to CSV\n",
        "sub_out_path = os.path.join(OUTPUT_DIR, 'submission.csv')\n",
        "submission.to_csv(sub_out_path, index=False)\n",
        "print(f\"Submission file created successfully at {sub_out_path}!\")\n",
        "print(submission['class'].value_counts())\n"
    ]
    
    data["cells"] = new_cells
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(f"Successfully updated {path}!")

if __name__ == "__main__":
    update_tuning_notebook()
