# Design Spec: 6-Model Hyperparameter Tuning Expansion

This document details the architecture and execution design for expanding our hyperparameter tuning pipeline to all 6 base estimators in the ensemble.

## Goal
Improve the out-of-fold and public leaderboard ensembling scores by replacing hardcoded baseline parameters for RandomForest (`rf`), ExtraTrees (`et`), and HistGradientBoosting (`hist_gb`) with optimal parameters found via GPU-accelerated Optuna searches.

## Proposed Architecture

### 1. Optuna Objective Additions
We will define three new objective functions in the tuning notebook:

```python
def tune_rf(trial):
    params = {
        'n_estimators': 300,
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    return evaluate_model(RandomForestClassifier(**params), 'rf')

def tune_et(trial):
    params = {
        'n_estimators': 300,
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    return evaluate_model(ExtraTreesClassifier(**params), 'et')

def tune_hist_gb(trial):
    params = {
        'max_iter': 300,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 12),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-8, 10.0, log=True),
        'class_weight': 'balanced',
        'random_state': 42
    }
    return evaluate_model(HistGradientBoostingClassifier(**params), 'hist_gb')
```

### 2. Output Export Format
The study loop will save the results as:
```json
{
    "lgb": { ... },
    "xgb": { ... },
    "cat": { ... },
    "rf": { ... },
    "et": { ... },
    "hist_gb": { ... }
}
```

### 3. Baseline Model Load Customization
The `get_model` function in `02_baseline_modeling.ipynb` will be updated to:
```python
    elif model_name == 'rf':
        p = tuned_params.get('rf', {})
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=p.get('max_depth', 12),
            min_samples_split=p.get('min_samples_split', 10),
            min_samples_leaf=p.get('min_samples_leaf', 5),
            class_weight='balanced',
            n_jobs=-1,
            random_state=seed
        )
    elif model_name == 'et':
        p = tuned_params.get('et', {})
        return ExtraTreesClassifier(
            n_estimators=500,
            max_depth=p.get('max_depth', 12),
            min_samples_split=p.get('min_samples_split', 10),
            min_samples_leaf=p.get('min_samples_leaf', 5),
            class_weight='balanced',
            n_jobs=-1,
            random_state=seed
        )
    elif model_name == 'hist_gb':
        p = tuned_params.get('hist_gb', {})
        return HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=p.get('learning_rate', 0.05),
            max_depth=p.get('max_depth', 8),
            min_samples_leaf=p.get('min_samples_leaf', 20),
            l2_regularization=p.get('l2_regularization', 1.0),
            class_weight='balanced',
            random_state=seed
        )
```

## Execution Verification Plan
1. Validate JSON syntax of the modified notebooks.
2. Push tuning notebook to Kaggle and verify execution using GPU accelerator.
3. Push baseline and ensembling runs and check final metrics.
