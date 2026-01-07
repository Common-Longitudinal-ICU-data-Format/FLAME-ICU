import marimo

__generated_with = "0.16.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # FLAIR Model Training and Evaluation

    This notebook trains XGBoost and ElasticNet models for tasks 5, 6, and 7.

    ## Prerequisites
    1. Run `01_task_generator.py` to generate task datasets
    2. Run `02_feature_engineering.py` to extract features

    ## Models
    - **XGBoost**: Gradient boosting with optimized hyperparameters
    - **ElasticNet**: L1+L2 regularized linear model

    ## Tasks
    - **Task 5 (ICU LOS)**: Regression - predict ICU length of stay
    - **Task 6 (Hospital Mortality)**: Binary classification - predict mortality
    - **Task 7 (ICU Readmission)**: Binary classification - predict readmission
    """
    )
    return


@app.cell
def _():
    import sys
    sys.path.append('../..')

    import pandas as pd
    import numpy as np
    from pathlib import Path
    import json
    import warnings
    warnings.filterwarnings('ignore')

    from code.models.xgboost_model import XGBoostModel
    from code.models.elasticnet_model import ElasticNetModel
    from code.models.evaluation import (
        TaskEvaluator, evaluate_task5, evaluate_task6, evaluate_task7, print_metrics
    )

    print("=== FLAIR Model Training and Evaluation ===")
    return (
        ElasticNetModel,
        Path,
        TaskEvaluator,
        XGBoostModel,
        evaluate_task5,
        evaluate_task6,
        evaluate_task7,
        json,
        np,
        pd,
        print_metrics,
        sys,
        warnings,
    )


@app.cell
def _(Path):
    # Configuration
    FEATURES_DIR = Path("../../outputs/features")
    MODELS_DIR = Path("../../outputs/models")
    RESULTS_DIR = Path("../../outputs/results")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Task configuration
    TASK_CONFIG = {
        'task5_icu_los': {
            'type': 'regression',
            'label_col': 'icu_los_hours',
            'display_name': 'ICU Length of Stay'
        },
        'task6_hospital_mortality': {
            'type': 'classification',
            'label_col': 'label_mortality',
            'display_name': 'Hospital Mortality'
        },
        'task7_icu_readmission': {
            'type': 'classification',
            'label_col': 'label_icu_readmission',
            'display_name': 'ICU Readmission'
        }
    }

    print(f"Configuration:")
    print(f"  Features dir: {FEATURES_DIR}")
    print(f"  Models dir: {MODELS_DIR}")
    print(f"  Results dir: {RESULTS_DIR}")
    return FEATURES_DIR, MODELS_DIR, RESULTS_DIR, TASK_CONFIG


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Features""")
    return


@app.cell
def _(FEATURES_DIR, TASK_CONFIG, pd):
    # Load feature datasets
    datasets = {}

    for task_name, config in TASK_CONFIG.items():
        features_path = FEATURES_DIR / f"{task_name}_final.parquet"
        if features_path.exists():
            df = pd.read_parquet(features_path)
            datasets[task_name] = df
            print(f"Loaded {task_name}: {df.shape}")
        else:
            print(f"WARNING: {task_name} features not found at {features_path}")
            print("  Please run 02_feature_engineering.py first")

    print(f"\nLoaded {len(datasets)} datasets")
    return config, datasets, df, features_path, task_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Prepare Train/Test Splits""")
    return


@app.cell
def _(TASK_CONFIG, datasets):
    # Prepare train/test data for each task
    prepared_data = {}

    for task_name, df in datasets.items():
        config = TASK_CONFIG[task_name]
        label_col = config['label_col']

        # Split data
        train_df = df[df['split'] == 'train'].copy()
        test_df = df[df['split'] == 'test'].copy()

        # Feature columns (exclude ID, split, labels)
        exclude_cols = ['hospitalization_id', 'split', 'icu_los_hours', 'label_mortality', 'label_icu_readmission']
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Prepare X, y
        X_train = train_df[feature_cols]
        y_train = train_df[label_col].values
        X_test = test_df[feature_cols]
        y_test = test_df[label_col].values

        prepared_data[task_name] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'feature_cols': feature_cols,
            'config': config
        }

        print(f"\n{task_name}:")
        print(f"  Train: {X_train.shape}, positive rate: {y_train.mean():.3f}")
        print(f"  Test: {X_test.shape}, positive rate: {y_test.mean():.3f}")
        print(f"  Features: {len(feature_cols)}")

    return (
        X_test,
        X_train,
        config,
        df,
        exclude_cols,
        feature_cols,
        label_col,
        prepared_data,
        task_name,
        test_df,
        train_df,
        y_test,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Train Models""")
    return


@app.cell
def _(ElasticNetModel, MODELS_DIR, XGBoostModel, prepared_data):
    # Train XGBoost and ElasticNet for each task
    models = {}
    all_results = {}

    for task_name, data in prepared_data.items():
        print(f"\n{'='*60}")
        print(f"Training models for: {task_name}")
        print(f"{'='*60}")

        task_type = data['config']['type']
        X_train = data['X_train']
        y_train = data['y_train']

        # Initialize models dict for this task
        models[task_name] = {}

        # Train XGBoost
        print(f"\n--- XGBoost ---")
        xgb_model = XGBoostModel(task_type=task_type)
        xgb_model.train(X_train, y_train, verbose=True)

        # Save XGBoost model
        xgb_path = MODELS_DIR / f"{task_name}_xgboost.json"
        xgb_model.save(str(xgb_path))
        models[task_name]['xgboost'] = xgb_model

        # Train ElasticNet
        print(f"\n--- ElasticNet ---")
        en_model = ElasticNetModel(task_type=task_type)
        en_model.train(X_train, y_train, verbose=True)

        # Save ElasticNet model
        en_path = MODELS_DIR / f"{task_name}_elasticnet.joblib"
        en_model.save(str(en_path))
        models[task_name]['elasticnet'] = en_model

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")
    return (
        X_train,
        all_results,
        data,
        en_model,
        en_path,
        models,
        task_name,
        task_type,
        xgb_model,
        xgb_path,
        y_train,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Evaluate Models""")
    return


@app.cell
def _(RESULTS_DIR, TaskEvaluator, json, models, np, prepared_data, print_metrics):
    # Evaluate all models on test set
    evaluation_results = {}

    for task_name, data in prepared_data.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {task_name}")
        print(f"{'='*60}")

        task_type = data['config']['type']
        X_test = data['X_test']
        y_test = data['y_test']

        evaluator = TaskEvaluator(task_type)
        evaluation_results[task_name] = {}

        for model_name, model in models[task_name].items():
            print(f"\n--- {model_name} ---")

            if task_type == 'regression':
                y_pred = model.predict(X_test)
                metrics = evaluator.evaluate(y_test, y_pred)
            else:
                y_pred_proba = model.predict_proba(X_test)
                y_pred_class = (y_pred_proba >= 0.5).astype(int)
                metrics = evaluator.evaluate(y_test, y_pred_class, y_pred_proba)

            evaluation_results[task_name][model_name] = metrics
            print_metrics(metrics, f"{task_name} - {model_name}")

    # Save results
    results_path = RESULTS_DIR / "evaluation_results.json"

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj

    results_json = convert_numpy(evaluation_results)

    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    return (
        X_test,
        convert_numpy,
        data,
        evaluation_results,
        evaluator,
        f,
        metrics,
        model,
        model_name,
        results_json,
        results_path,
        task_name,
        task_type,
        y_pred,
        y_pred_class,
        y_pred_proba,
        y_test,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Results Summary""")
    return


@app.cell
def _(TASK_CONFIG, evaluation_results, pd):
    # Create summary table
    summary_rows = []

    for task_name, task_results in evaluation_results.items():
        task_type = TASK_CONFIG[task_name]['type']
        display_name = TASK_CONFIG[task_name]['display_name']

        for model_name, metrics in task_results.items():
            row = {
                'Task': display_name,
                'Model': model_name.upper()
            }

            if task_type == 'regression':
                row['RMSE'] = f"{metrics.get('rmse', 0):.2f}"
                row['MAE'] = f"{metrics.get('mae', 0):.2f}"
                row['R2'] = f"{metrics.get('r2', 0):.4f}"
            else:
                row['AUROC'] = f"{metrics.get('auroc', 0):.4f}"
                row['AUPRC'] = f"{metrics.get('auprc', 0):.4f}"
                row['F1'] = f"{metrics.get('f1', 0):.4f}"

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    print("\n=== Results Summary ===\n")
    print(summary_df.to_string(index=False))
    return display_name, metrics, model_name, row, summary_df, summary_rows, task_name, task_results, task_type


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Feature Importance (XGBoost)""")
    return


@app.cell
def _(models):
    # Display top features for each XGBoost model
    for task_name, task_models in models.items():
        if 'xgboost' in task_models:
            print(f"\n=== {task_name} - Top 15 Features ===")
            importance_df = task_models['xgboost'].get_feature_importance('gain')
            print(importance_df.head(15).to_string(index=False))
    return importance_df, task_models, task_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Summary

    Models have been trained and evaluated for all three FLAIR benchmark tasks:
    - **Task 5**: ICU Length of Stay prediction (regression)
    - **Task 6**: Hospital Mortality prediction (classification)
    - **Task 7**: ICU Readmission prediction (classification)

    Results are saved in `outputs/results/evaluation_results.json`.
    Models are saved in `outputs/models/`.
    """
    )
    return


if __name__ == "__main__":
    app.run()
