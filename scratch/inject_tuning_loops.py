import json
import os

def inject_code():
    path = "notebooks/04_hyperparameter_tuning.ipynb"
    with open(path, "r", encoding="utf-8") as f:
        notebook = json.load(f)
        
    objective_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "tuning_logic",
        "metadata": {},
        "outputs": [],
        "source": [
            "def tune_lgb(trial):\n",
            "    params = {\n",
            "        'n_estimators': 800,\n",
            "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),\n",
            "        'num_leaves': trial.suggest_int('num_leaves', 31, 255),\n",
            "        'max_depth': trial.suggest_int('max_depth', 5, 12),\n",
            "        'subsample': trial.suggest_float('subsample', 0.6, 1.0),\n",
            "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),\n",
            "        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),\n",
            "        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),\n",
            "        'class_weight': 'balanced',\n",
            "        'random_state': 42,\n",
            "        'n_jobs': -1,\n",
            "        'verbose': -1\n",
            "    }\n",
            "    return evaluate_model(lgb.LGBMClassifier(**params), 'lgb')\n",
            "\n",
            "def tune_xgb(trial):\n",
            "    params = {\n",
            "        'n_estimators': 800,\n",
            "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),\n",
            "        'max_depth': trial.suggest_int('max_depth', 4, 10),\n",
            "        'subsample': trial.suggest_float('subsample', 0.6, 1.0),\n",
            "        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),\n",
            "        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),\n",
            "        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),\n",
            "        'tree_method': 'hist',\n",
            "        'device': 'cuda',\n",
            "        'random_state': 42,\n",
            "        'n_jobs': -1\n",
            "    }\n",
            "    return evaluate_model(xgb.XGBClassifier(**params), 'xgb')\n",
            "\n",
            "def tune_cat(trial):\n",
            "    params = {\n",
            "        'iterations': 800,\n",
            "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),\n",
            "        'depth': trial.suggest_int('depth', 4, 10),\n",
            "        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),\n",
            "        'random_strength': trial.suggest_float('random_strength', 1e-9, 10.0, log=True),\n",
            "        'auto_class_weights': 'Balanced',\n",
            "        'task_type': 'GPU',\n",
            "        'random_state': 42,\n",
            "        'verbose': False\n",
            "    }\n",
            "    return evaluate_model(CatBoostClassifier(**params), 'cat')\n",
            "\n",
            "def tune_rf(trial):\n",
            "    params = {\n",
            "        'n_estimators': 300,\n",
            "        'max_depth': trial.suggest_int('max_depth', 5, 15),\n",
            "        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),\n",
            "        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),\n",
            "        'class_weight': 'balanced',\n",
            "        'random_state': 42,\n",
            "        'n_jobs': -1\n",
            "    }\n",
            "    return evaluate_model(RandomForestClassifier(**params), 'rf')\n",
            "\n",
            "def tune_et(trial):\n",
            "    params = {\n",
            "        'n_estimators': 300,\n",
            "        'max_depth': trial.suggest_int('max_depth', 5, 15),\n",
            "        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),\n",
            "        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),\n",
            "        'class_weight': 'balanced',\n",
            "        'random_state': 42,\n",
            "        'n_jobs': -1\n",
            "    }\n",
            "    return evaluate_model(ExtraTreesClassifier(**params), 'et')\n",
            "\n",
            "def tune_hist_gb(trial):\n",
            "    params = {\n",
            "        'max_iter': 300,\n",
            "        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),\n",
            "        'max_depth': trial.suggest_int('max_depth', 4, 12),\n",
            "        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),\n",
            "        'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),\n",
            "        'class_weight': 'balanced',\n",
            "        'random_state': 42\n",
            "    }\n",
            "    return evaluate_model(HistGradientBoostingClassifier(**params), 'hist_gb')\n",
            "\n",
            "def evaluate_model(model, name):\n",
            "    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)\n",
            "    scores = []\n",
            "    for train_idx, val_idx in skf.split(X, y):\n",
            "        X_tr, y_tr = X.iloc[train_idx], y[train_idx]\n",
            "        X_va, y_va = X.iloc[val_idx], y[val_idx]\n",
            "        \n",
            "        if name == 'xgb':\n",
            "            sample_weight = compute_sample_weight('balanced', y_tr)\n",
            "            model.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)\n",
            "        else:\n",
            "            model.fit(X_tr, y_tr)\n",
            "            \n",
            "        preds = model.predict(X_va)\n",
            "        if hasattr(preds, 'ndim') and preds.ndim > 1:\n",
            "            preds = preds.squeeze()\n",
            "        scores.append(balanced_accuracy_score(y_va, preds))\n",
            "    return np.mean(scores)\n"
        ]
    }

    run_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "run_optuna",
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Starting Optuna Studies...\")\n",
            "optuna.logging.set_verbosity(optuna.logging.WARNING)\n",
            "\n",
            "study_lgb = optuna.create_study(direction='maximize')\n",
            "study_lgb.optimize(tune_lgb, n_trials=30)\n",
            "print(f\"Best LGB Score: {study_lgb.best_value:.5f}\")\n",
            "\n",
            "study_xgb = optuna.create_study(direction='maximize')\n",
            "study_xgb.optimize(tune_xgb, n_trials=30)\n",
            "print(f\"Best XGB Score: {study_xgb.best_value:.5f}\")\n",
            "\n",
            "study_cat = optuna.create_study(direction='maximize')\n",
            "study_cat.optimize(tune_cat, n_trials=30)\n",
            "print(f\"Best Cat Score: {study_cat.best_value:.5f}\")\n",
            "\n",
            "study_rf = optuna.create_study(direction='maximize')\n",
            "study_rf.optimize(tune_rf, n_trials=30)\n",
            "print(f\"Best RF Score: {study_rf.best_value:.5f}\")\n",
            "\n",
            "study_et = optuna.create_study(direction='maximize')\n",
            "study_et.optimize(tune_et, n_trials=30)\n",
            "print(f\"Best ET Score: {study_et.best_value:.5f}\")\n",
            "\n",
            "study_hist = optuna.create_study(direction='maximize')\n",
            "study_hist.optimize(tune_hist_gb, n_trials=30)\n",
            "print(f\"Best HistGB Score: {study_hist.best_value:.5f}\")\n",
            "\n",
            "# Export optimal hyperparameters\n",
            "best_params = {\n",
            "    'lgb': study_lgb.best_params,\n",
            "    'xgb': study_xgb.best_params,\n",
            "    'cat': study_cat.best_params,\n",
            "    'rf': study_rf.best_params,\n",
            "    'et': study_et.best_params,\n",
            "    'hist_gb': study_hist.best_params\n",
            "}\n",
            "\n",
            "out_path = os.path.join(OUTPUT_DIR, 'best_hyperparameters.json')\n",
            "with open(out_path, 'w', encoding='utf-8') as f:\n",
            "    json.dump(best_params, f, indent=4)\n",
            "print(f\"Saved best parameters to {out_path}!\")\n"
        ]
    }
    
    # Clean old inject logic
    notebook["cells"] = notebook["cells"][:3]
    
    # Modify Cell 1 (imports cell) to include the sklearn ensemble imports
    imports_cell = notebook["cells"][1]
    import_stmt = "from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier\n"
    if not any("RandomForestClassifier" in line for line in imports_cell["source"]):
        if imports_cell["source"] and not imports_cell["source"][-1].endswith("\n"):
            imports_cell["source"][-1] += "\n"
        imports_cell["source"].append(import_stmt)

    notebook["cells"].extend([objective_cell, run_cell])
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print("Updated hyperparameter tuning notebook layout.")

if __name__ == "__main__":
    inject_code()
