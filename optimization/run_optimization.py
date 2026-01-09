#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization Script

This script optimizes hyperparameters for XGBoost and ElasticNet models
for Task 5 (ICU LOS - regression) and Task 7 (ICU Readmission - classification).

Usage:
    python run_optimization.py                          # Run all optimizations
    python run_optimization.py --task task7_icu_readmission --model xgboost
    python run_optimization.py --n_trials 100           # More trials for better results
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
RESULTS_DIR = Path(__file__).parent / "results"

TASKS = {
    "task5_icu_los": {
        "type": "regression",
        "metric": "r2",
        "sklearn_scoring": "r2",
        "label": "icu_los_hours",
        "feature_file": "task5_icu_los_final.parquet",
        "direction": "maximize",
    },
    "task7_icu_readmission": {
        "type": "classification",
        "metric": "auroc",
        "sklearn_scoring": "roc_auc",
        "label": "label_icu_readmission",
        "feature_file": "task7_icu_readmission_final.parquet",
        "direction": "maximize",
    },
}

MODELS = ["xgboost", "elasticnet"]

# Default settings
DEFAULT_N_TRIALS = 50
DEFAULT_CV_FOLDS = 3
RANDOM_STATE = 42


# =============================================================================
# Data Loading
# =============================================================================


def load_task_data(task_name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load feature data for a given task."""
    task_config = TASKS[task_name]
    feature_file = FEATURES_DIR / task_config["feature_file"]

    if not feature_file.exists():
        raise FileNotFoundError(f"Feature file not found: {feature_file}")

    df = pd.read_parquet(feature_file)

    # Get label column
    label_col = task_config["label"]
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {feature_file}")

    # Separate features and labels
    y = df[label_col].copy()

    # Drop non-feature columns
    drop_cols = [
        label_col,
        "hospitalization_id",
        "patient_id",
        "admission_dttm",
        "discharge_dttm",
        "icu_admission_dttm",
        "icu_discharge_dttm",
        "split",
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Drop any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])

    # Remove rows with missing labels
    valid_mask = ~y.isna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    print(f"  Loaded {len(X)} samples with {X.shape[1]} features")
    print(f"  Label distribution: {y.value_counts().to_dict() if task_config['type'] == 'classification' else f'mean={y.mean():.2f}, std={y.std():.2f}'}")

    return X, y


# =============================================================================
# XGBoost Optimization
# =============================================================================


def create_xgboost_objective(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    scoring: str,
    cv_folds: int,
):
    """Create Optuna objective function for XGBoost."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0,
        }

        if task_type == "classification":
            # Calculate scale_pos_weight for class imbalance
            n_neg = (y == 0).sum()
            n_pos = (y == 1).sum()
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            params["scale_pos_weight"] = scale_pos_weight
            params["eval_metric"] = "logloss"

            model = XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        else:
            params["eval_metric"] = "rmse"
            model = XGBRegressor(**params)
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

        # Cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return scores.mean()

    return objective


# =============================================================================
# ElasticNet Optimization
# =============================================================================


def create_elasticnet_objective(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    scoring: str,
    cv_folds: int,
):
    """Create Optuna objective function for ElasticNet."""

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-4, 10, log=True)
        l1_ratio = trial.suggest_float("l1_ratio", 0.0, 1.0)
        max_iter = trial.suggest_int("max_iter", 1000, 5000)

        # Create pipeline with imputation and scaling
        if task_type == "classification":
            # LogisticRegression with elastic net penalty
            # C = 1/alpha (inverse regularization strength)
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    C=1.0 / alpha,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                )),
            ])
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        else:
            # ElasticNet for regression
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                    random_state=RANDOM_STATE,
                )),
            ])
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

        # Cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        return scores.mean()

    return objective


# =============================================================================
# Optimization Runner
# =============================================================================


def run_optimization(
    task_name: str,
    model_name: str,
    n_trials: int,
    cv_folds: int,
) -> dict:
    """Run Optuna optimization for a specific task and model."""

    print(f"\n{'='*60}")
    print(f"Optimizing {model_name.upper()} for {task_name}")
    print(f"{'='*60}")

    task_config = TASKS[task_name]
    task_type = task_config["type"]
    scoring = task_config["sklearn_scoring"]
    direction = task_config["direction"]

    # Load data
    print("\nLoading data...")
    X, y = load_task_data(task_name)

    # Create objective function
    if model_name == "xgboost":
        objective = create_xgboost_objective(X, y, task_type, scoring, cv_folds)
    else:
        objective = create_elasticnet_objective(X, y, task_type, scoring, cv_folds)

    # Create study with pruning
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
    )

    # Optimize with progress callback
    print(f"\nRunning {n_trials} trials with {cv_folds}-fold CV...")

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if trial.number % 10 == 0 or trial.number == n_trials - 1:
            print(f"  Trial {trial.number + 1}/{n_trials}: {trial.value:.4f} (best: {study.best_value:.4f})")

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=[callback],
        show_progress_bar=True,
    )

    # Prepare results
    results = {
        "task": task_name,
        "model": model_name,
        "task_type": task_type,
        "metric": task_config["metric"],
        "best_score": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
        "cv_folds": cv_folds,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{task_name}_{model_name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Print best parameters
    print(f"\nBest {task_config['metric'].upper()}: {study.best_value:.4f}")
    print("Best parameters:")
    for param, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")

    return results


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimization for FLAIR Benchmark"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=list(TASKS.keys()),
        default=None,
        help="Task to optimize (default: all tasks)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=MODELS,
        default=None,
        help="Model to optimize (default: all models)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f"Number of Optuna trials (default: {DEFAULT_N_TRIALS})",
    )
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=DEFAULT_CV_FOLDS,
        help=f"Number of CV folds (default: {DEFAULT_CV_FOLDS})",
    )
    args = parser.parse_args()

    # Determine tasks and models to optimize
    tasks = [args.task] if args.task else list(TASKS.keys())
    models = [args.model] if args.model else MODELS

    print("=" * 60)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    print(f"Tasks: {tasks}")
    print(f"Models: {models}")
    print(f"Trials per combination: {args.n_trials}")
    print(f"CV folds: {args.cv_folds}")

    # Run optimizations
    all_results = []
    for task_name in tasks:
        for model_name in models:
            try:
                results = run_optimization(
                    task_name=task_name,
                    model_name=model_name,
                    n_trials=args.n_trials,
                    cv_folds=args.cv_folds,
                )
                all_results.append(results)
            except Exception as e:
                print(f"\nERROR: Failed to optimize {model_name} for {task_name}")
                print(f"  {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"\n{'Task':<30} {'Model':<12} {'Metric':<8} {'Score':<10}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['task']:<30} {r['model']:<12} {r['metric']:<8} {r['best_score']:.4f}")

    print(f"\nResults saved to: {RESULTS_DIR}")
    print("\nTo use these parameters in training, update the model config files")
    print("or pass them directly to the model constructors.")


if __name__ == "__main__":
    main()
