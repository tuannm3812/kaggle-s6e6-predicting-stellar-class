import json
import os

def update_baseline_notebook():
    path = "notebooks/02_baseline_modeling.ipynb"
    print(f"Reverting {path} to git version to ensure idempotency...")
    os.system(f"git checkout {path}")
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 1. Update imports (cell index 1)
    cell_1 = data["cells"][1]
    cell_1["source"] = [
        "import json\n",
        "import os\n",
        "import numpy as np\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import StratifiedKFold\n",
        "from sklearn.metrics import balanced_accuracy_score\n",
        "from sklearn.utils.class_weight import compute_sample_weight\n",
        "from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier\n",
        "import lightgbm as lgb\n",
        "import xgboost as xgb\n",
        "from catboost import CatBoostClassifier"
    ]
    
    # 2. Update get_model (cell index 7) to include new models
    cell_7 = data["cells"][7]
    cell_7["source"] = [
        "def get_model(model_name: str, seed: int = 42):\n",
        "    \"\"\"Initializes the classifier model using tuned hyperparameters if available.\n",
        "    \"\"\"\n",
        "    # Check for tuned hyperparameters JSON\n",
        "    params_path = 'best_hyperparameters.json'\n",
        "    if not os.path.exists(params_path):\n",
        "        # fallback for Kaggle inputs folder\n",
        "        params_path = '/kaggle/input/stellar-classification-hyperparameter-tuning/best_hyperparameters.json'\n",
        "        if not os.path.exists(params_path):\n",
        "            params_path = '../scratch/best_hyperparameters.json'\n",
        "            \n",
        "    tuned_params = {}\n",
        "    if os.path.exists(params_path):\n",
        "        try:\n",
        "            with open(params_path, 'r') as f:\n",
        "                tuned_params = json.load(f)\n",
        "            print(f\"Loaded tuned hyperparameters from {params_path}!\")\n",
        "        except Exception as e:\n",
        "            print(f\"Error loading tuned parameters: {e}\")\n",
        "            \n",
        "    if model_name == 'lgb':\n",
        "        p = tuned_params.get('lgb', {})\n",
        "        return lgb.LGBMClassifier(\n",
        "            n_estimators=1000,\n",
        "            learning_rate=p.get('learning_rate', 0.05),\n",
        "            num_leaves=p.get('num_leaves', 63),\n",
        "            max_depth=p.get('max_depth', 8),\n",
        "            subsample=p.get('subsample', 0.8),\n",
        "            colsample_bytree=p.get('colsample_bytree', 0.8),\n",
        "            reg_alpha=p.get('reg_alpha', 1e-8),\n",
        "            reg_lambda=p.get('reg_lambda', 1e-8),\n",
        "            class_weight='balanced',\n",
        "            random_state=seed,\n",
        "            n_jobs=-1,\n",
        "            verbose=-1\n",
        "        )\n",
        "    elif model_name == 'xgb':\n",
        "        p = tuned_params.get('xgb', {})\n",
        "        return xgb.XGBClassifier(\n",
        "            n_estimators=1000,\n",
        "            learning_rate=p.get('learning_rate', 0.05),\n",
        "            max_depth=p.get('max_depth', 7),\n",
        "            subsample=p.get('subsample', 0.8),\n",
        "            colsample_bytree=p.get('colsample_bytree', 0.8),\n",
        "            reg_alpha=p.get('reg_alpha', 1e-8),\n",
        "            reg_lambda=p.get('reg_lambda', 1e-8),\n",
        "            tree_method='hist',\n",
        "            random_state=seed,\n",
        "            n_jobs=-1\n",
        "        )\n",
        "    elif model_name == 'cat':\n",
        "        p = tuned_params.get('cat', {})\n",
        "        return CatBoostClassifier(\n",
        "            iterations=1000,\n",
        "            learning_rate=p.get('learning_rate', 0.05),\n",
        "            depth=p.get('depth', 7),\n",
        "            l2_leaf_reg=p.get('l2_leaf_reg', 3.0),\n",
        "            random_strength=p.get('random_strength', 1e-9),\n",
        "            auto_class_weights='Balanced',\n",
        "            random_state=seed,\n",
        "            thread_count=-1,\n",
        "            verbose=False\n",
        "        )\n",
        "    elif model_name == 'rf':\n",
        "        p = tuned_params.get('rf', {})\n",
        "        return RandomForestClassifier(\n",
        "            n_estimators=500,\n",
        "            max_depth=p.get('max_depth', 12),\n",
        "            min_samples_split=p.get('min_samples_split', 10),\n",
        "            min_samples_leaf=p.get('min_samples_leaf', 5),\n",
        "            class_weight='balanced',\n",
        "            n_jobs=-1,\n",
        "            random_state=seed\n",
        "        )\n",
        "    elif model_name == 'et':\n",
        "        p = tuned_params.get('et', {})\n",
        "        return ExtraTreesClassifier(\n",
        "            n_estimators=500,\n",
        "            max_depth=p.get('max_depth', 12),\n",
        "            min_samples_split=p.get('min_samples_split', 10),\n",
        "            min_samples_leaf=p.get('min_samples_leaf', 5),\n",
        "            class_weight='balanced',\n",
        "            n_jobs=-1,\n",
        "            random_state=seed\n",
        "        )\n",
        "    elif model_name == 'hist_gb':\n",
        "        p = tuned_params.get('hist_gb', {})\n",
        "        return HistGradientBoostingClassifier(\n",
        "            max_iter=500,\n",
        "            learning_rate=p.get('learning_rate', 0.05),\n",
        "            max_depth=p.get('max_depth', 8),\n",
        "            min_samples_leaf=p.get('min_samples_leaf', 20),\n",
        "            l2_regularization=p.get('l2_regularization', 1.0),\n",
        "            class_weight='balanced',\n",
        "            random_state=seed\n",
        "        )\n",
        "    else:\n",
        "        raise ValueError(f\"Unknown model: {model_name}\")\n",
        "\n",
        "def train_cv(model_name: str, n_splits: int = 5, seed: int = 42) -> tuple[np.ndarray, np.ndarray, float]:\n",
        "    \"\"\"Trains a model using Stratified K-Fold CV.\n",
        "    \"\"\"\n",
        "    print(f\"\\nTraining {model_name.upper()} model using {n_splits}-fold Stratified CV...\")\n",
        "    \n",
        "    global train_raw_global, test_raw_global\n",
        "    train_raw = train_raw_global\n",
        "    test_raw = test_raw_global\n",
        "    \n",
        "    X = feature_engineering(train_raw)\n",
        "    y = X.pop('class').map(TARGET_MAP)\n",
        "    X_test = feature_engineering(test_raw)\n",
        "    \n",
        "    print(f\"Feature columns: {list(X.columns)}\")\n",
        "    print(f\"Training shape: {X.shape}, Test shape: {X_test.shape}\")\n",
        "    \n",
        "    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)\n",
        "    \n",
        "    oof_preds = np.zeros((len(X), 3))\n",
        "    test_preds = np.zeros((len(X_test), 3))\n",
        "    fold_scores = []\n",
        "    \n",
        "    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):\n",
        "        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]\n",
        "        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]\n",
        "        \n",
        "        model = get_model(model_name, seed)\n",
        "        \n",
        "        if model_name == 'lgb':\n",
        "            model.fit(\n",
        "                X_train, y_train,\n",
        "                eval_set=[(X_val, y_val)],\n",
        "                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]\n",
        "            )\n",
        "        elif model_name == 'xgb':\n",
        "            sample_weight = compute_sample_weight('balanced', y_train)\n",
        "            model.fit(\n",
        "                X_train, y_train,\n",
        "                sample_weight=sample_weight,\n",
        "                eval_set=[(X_val, y_val)],\n",
        "                verbose=False\n",
        "            )\n",
        "        elif model_name == 'cat':\n",
        "            model.fit(\n",
        "                X_train, y_train,\n",
        "                eval_set=(X_val, y_val),\n",
        "                early_stopping_rounds=50,\n",
        "                verbose=False\n",
        "            )\n",
        "        else:\n",
        "            # RandomForest, ExtraTrees, HistGradientBoosting do not support eval_set/early stopping\n",
        "            model.fit(X_train, y_train)\n",
        "            \n",
        "        val_preds_prob = model.predict_proba(X_val)\n",
        "        oof_preds[val_idx] = val_preds_prob\n",
        "        \n",
        "        val_preds = np.argmax(val_preds_prob, axis=1)\n",
        "        score = balanced_accuracy_score(y_val, val_preds)\n",
        "        fold_scores.append(score)\n",
        "        print(f\"Fold {fold+1} Balanced Accuracy: {score:.5f}\")\n",
        "        \n",
        "        test_preds += model.predict_proba(X_test) / n_splits\n",
        "        \n",
        "    cv_score = balanced_accuracy_score(y, np.argmax(oof_preds, axis=1))\n",
        "    print(f\"\\n{model_name.upper()} CV Balanced Accuracy Score: {cv_score:.5f}\")\n",
        "    \n",
        "    os.makedirs(PRED_DIR, exist_ok=True)\n",
        "    np.save(os.path.join(PRED_DIR, f'{model_name}_oof.npy'), oof_preds)\n",
        "    np.save(os.path.join(PRED_DIR, f'{model_name}_test.npy'), test_preds)\n",
        "    \n",
        "    return oof_preds, test_preds, cv_score"
    ]
    
    # 3. Inject new cells for RF, ET, and HistGB
    data["cells"][9]["source"] = ["lgb_oof, lgb_test, lgb_score = train_cv('lgb', n_splits=N_SPLITS)"]
    data["cells"][10]["source"] = ["xgb_oof, xgb_test, xgb_score = train_cv('xgb', n_splits=N_SPLITS)"]
    data["cells"][11]["source"] = ["cat_oof, cat_test, cat_score = train_cv('cat', n_splits=N_SPLITS)"]
    
    rf_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "rf_train_run",
        "metadata": {},
        "outputs": [],
        "source": ["rf_oof, rf_test, rf_score = train_cv('rf', n_splits=N_SPLITS)"]
    }
    et_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "et_train_run",
        "metadata": {},
        "outputs": [],
        "source": ["et_oof, et_test, et_score = train_cv('et', n_splits=N_SPLITS)"]
    }
    hist_gb_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "hist_gb_train_run",
        "metadata": {},
        "outputs": [],
        "source": ["hist_gb_oof, hist_gb_test, hist_gb_score = train_cv('hist_gb', n_splits=N_SPLITS)"]
    }
    
    # Only inject cells if they haven't been injected yet to ensure idempotency
    has_rf = any(c.get("id") == "rf_train_run" for c in data["cells"])
    if not has_rf:
        new_cells = data["cells"][:12] + [rf_cell, et_cell, hist_gb_cell] + data["cells"][12:]
        data["cells"] = new_cells
    else:
        # If already injected, update them in place to ensure they are clean
        for c in data["cells"]:
            if c.get("id") == "rf_train_run":
                c["source"] = rf_cell["source"]
            elif c.get("id") == "et_train_run":
                c["source"] = et_cell["source"]
            elif c.get("id") == "hist_gb_train_run":
                c["source"] = hist_gb_cell["source"]
    
    # 4. Update the summary printing cell (which will be at index 16 after injecting cells)
    data["cells"][16]["source"] = [
        "print(\"Baseline CV Scores (Balanced Accuracy):\")\n",
        "print(f\"LightGBM: {lgb_score:.5f}\")\n",
        "print(f\"XGBoost:  {xgb_score:.5f}\")\n",
        "print(f\"CatBoost: {cat_score:.5f}\")\n",
        "print(f\"RF:       {rf_score:.5f}\")\n",
        "print(f\"ET:       {et_score:.5f}\")\n",
        "print(f\"HistGB:   {hist_gb_score:.5f}\")"
    ]
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(f"Successfully updated {path}!")

if __name__ == "__main__":
    update_baseline_notebook()
