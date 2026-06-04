import json
import os

def update_baseline():
    path = "notebooks/02_baseline_modeling.ipynb"
    print(f"Loading {path}...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    cell = data["cells"][7]
    assert "def get_model" in "".join(cell["source"])
    
    # We replace get_model implementation to read best_hyperparameters.json
    cell["source"] = [
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
        "    else:\n",
        "        raise ValueError(f\"Unknown model: {model_name}\")"
    ]
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    print(f"Successfully integrated tuned parameters into {path}!")

if __name__ == "__main__":
    update_baseline()
