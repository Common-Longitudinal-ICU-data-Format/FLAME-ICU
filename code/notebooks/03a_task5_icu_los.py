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
    # Task 5: ICU Length of Stay Prediction

    **Task Type:** Regression
    **Label:** `icu_los_hours`
    **Metrics:** MSE, RMSE, MAE, R²

    ## Site-Aware Training Logic
    - **If site == "rush":** Train models and save to `rush_models/`
    - **If site != "rush":** Run 3 experiments:
      1. Rush Model Evaluation (test Rush models on local data)
      2. Transfer Learning (fine-tune Rush models with local data)
      3. Independent Training (train from scratch)
    """
    )
    return


@app.cell
def _():
    import sys
    import os
    import json
    from pathlib import Path
    from datetime import datetime

    import pandas as pd
    import numpy as np

    # Determine project root
    _cwd = Path(os.getcwd()).resolve()
    if _cwd.name == 'notebooks':
        PROJECT_ROOT = _cwd.parent.parent
    elif _cwd.name == 'code':
        PROJECT_ROOT = _cwd.parent
    else:
        PROJECT_ROOT = _cwd

    # Add project root to path
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    # Import using importlib to avoid 'code' module conflict
    import importlib.util
    _spec_xgb = importlib.util.spec_from_file_location(
        "xgboost_model", PROJECT_ROOT / "code" / "models" / "xgboost_model.py"
    )
    _xgb_module = importlib.util.module_from_spec(_spec_xgb)
    _spec_xgb.loader.exec_module(_xgb_module)
    XGBoostModel = _xgb_module.XGBoostModel

    _spec_en = importlib.util.spec_from_file_location(
        "elasticnet_model", PROJECT_ROOT / "code" / "models" / "elasticnet_model.py"
    )
    _en_module = importlib.util.module_from_spec(_spec_en)
    _spec_en.loader.exec_module(_en_module)
    ElasticNetModel = _en_module.ElasticNetModel

    _spec_eval = importlib.util.spec_from_file_location(
        "evaluation", PROJECT_ROOT / "code" / "models" / "evaluation.py"
    )
    _eval_module = importlib.util.module_from_spec(_spec_eval)
    _spec_eval.loader.exec_module(_eval_module)
    TaskEvaluator = _eval_module.TaskEvaluator
    print_metrics = _eval_module.print_metrics
    plot_predicted_vs_observed = _eval_module.plot_predicted_vs_observed

    print("=== Task 5: ICU Length of Stay ===")
    return (
        ElasticNetModel, Path, PROJECT_ROOT, TaskEvaluator, XGBoostModel,
        datetime, importlib, json, np, os, pd, plot_predicted_vs_observed, print_metrics, sys
    )


@app.cell
def _(PROJECT_ROOT, Path, json):
    # Configuration
    CONFIG_PATH = PROJECT_ROOT / "clif_config.json"
    FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
    RUSH_MODELS_DIR = PROJECT_ROOT / "rush_models"
    RESULTS_DIR = PROJECT_ROOT / "results_to_box"

    # Task configuration
    TASK_NAME = "task5_icu_los"
    TASK_TYPE = "regression"
    LABEL_COL = "icu_los_hours"

    # Read site from config
    with open(CONFIG_PATH) as _f:
        _config = json.load(_f)
    SITE_NAME = _config.get('site', 'unknown').lower()
    IS_RUSH = (SITE_NAME == 'rush')

    print(f"Configuration:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Site: {SITE_NAME}")
    print(f"  Is Rush: {IS_RUSH}")
    print(f"  Task: {TASK_NAME} ({TASK_TYPE})")
    print(f"  Label column: {LABEL_COL}")
    return (
        CONFIG_PATH, FEATURES_DIR, IS_RUSH, LABEL_COL, RESULTS_DIR,
        RUSH_MODELS_DIR, SITE_NAME, TASK_NAME, TASK_TYPE
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Features""")
    return


@app.cell
def _(FEATURES_DIR, LABEL_COL, TASK_NAME, pd):
    # Load task features
    _features_path = FEATURES_DIR / f"{TASK_NAME}_final.parquet"
    features_df = pd.read_parquet(_features_path)

    print(f"Loaded features: {features_df.shape}")
    print(f"  Train: {len(features_df[features_df['split'] == 'train'])}")
    print(f"  Test: {len(features_df[features_df['split'] == 'test'])}")

    # Check label column
    if LABEL_COL in features_df.columns:
        print(f"  Label stats: mean={features_df[LABEL_COL].mean():.2f}, median={features_df[LABEL_COL].median():.2f}")
    else:
        print(f"  WARNING: Label column '{LABEL_COL}' not found!")
    return (features_df,)


@app.cell
def _(LABEL_COL, PROJECT_ROOT, features_df, json, np):
    # Load feature config from JSON
    with open(PROJECT_ROOT / "code" / "feature_config.json") as _f:
        _feature_config = json.load(_f)

    # Get model features that exist in the data
    _feature_cols = [c for c in _feature_config['model_features'] if c in features_df.columns]
    _missing_features = [c for c in _feature_config['model_features'] if c not in features_df.columns]

    if _missing_features:
        print(f"  Warning: Missing features from config: {_missing_features}")

    print(f"  Using {len(_feature_cols)}/{len(_feature_config['model_features'])} features from feature_config.json")

    _train_df = features_df[features_df['split'] == 'train']
    _test_df = features_df[features_df['split'] == 'test']

    X_train = _train_df[_feature_cols]
    y_train = _train_df[LABEL_COL].values
    X_test = _test_df[_feature_cols]
    y_test = _test_df[LABEL_COL].values

    print(f"Data prepared:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  Features: {len(_feature_cols)}")
    print(f"  Missing values in X_train: {X_train.isnull().sum().sum()}")
    return X_test, X_train, y_test, y_train


@app.cell(hide_code=True)
def _(IS_RUSH, mo):
    mo.md(
        f"""
    ## {'Rush Site: Train & Save Models' if IS_RUSH else 'Non-Rush Site: Run 3 Experiments'}

    {'Training XGBoost and ElasticNet models, saving to `rush_models/`' if IS_RUSH else 'Running Rush evaluation, transfer learning, and independent training'}
    """
    )
    return


@app.cell
def _(
    ElasticNetModel, IS_RUSH, RUSH_MODELS_DIR, TASK_NAME, TASK_TYPE,
    TaskEvaluator, X_test, X_train, XGBoostModel, datetime, json, np,
    plot_predicted_vs_observed, print_metrics, y_test, y_train
):
    # Initialize results storage
    all_results = {}

    if IS_RUSH:
        # ============================================================
        # RUSH SITE: Train and save models
        # ============================================================
        print("\n" + "="*60)
        print("RUSH SITE: Training and saving models")
        print("="*60)

        RUSH_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        evaluator = TaskEvaluator(TASK_TYPE)

        # --- Train XGBoost ---
        print("\n--- Training XGBoost ---")
        xgb_model = XGBoostModel(task_type=TASK_TYPE)
        xgb_model.train(X_train, y_train, verbose=True)

        # Evaluate
        y_pred_xgb = xgb_model.predict(X_test)
        metrics_xgb = evaluator.evaluate(y_test, y_pred_xgb)
        ci_xgb = evaluator.compute_bootstrap_ci(y_test, y_pred_xgb, n_iterations=1000)
        print_metrics(metrics_xgb, "XGBoost Test Metrics")

        # Save model
        xgb_model.save(str(RUSH_MODELS_DIR / f"{TASK_NAME}_xgboost.json"))

        # --- Train ElasticNet ---
        print("\n--- Training ElasticNet ---")
        en_model = ElasticNetModel(task_type=TASK_TYPE)
        en_model.train(X_train, y_train, verbose=True)

        # Evaluate
        y_pred_en = en_model.predict(X_test)
        metrics_en = evaluator.evaluate(y_test, y_pred_en)
        ci_en = evaluator.compute_bootstrap_ci(y_test, y_pred_en, n_iterations=1000)
        print_metrics(metrics_en, "ElasticNet Test Metrics")

        # Save model
        en_model.save(str(RUSH_MODELS_DIR / f"{TASK_NAME}_elasticnet.joblib"))

        # Save metrics
        _rush_metrics = {
            "site": "rush",
            "task": TASK_NAME,
            "experiment": "rush_training",
            "timestamp": datetime.now().isoformat(),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "models": {
                "xgboost": ci_xgb,
                "elasticnet": ci_en
            }
        }

        _metrics_path = RUSH_MODELS_DIR / f"{TASK_NAME}_metrics.json"
        with open(_metrics_path, 'w') as _f:
            json.dump(_rush_metrics, _f, indent=2)

        # Structure for summary display
        all_results = {
            "rush_training": _rush_metrics
        }
        print(f"\nMetrics saved to: {_metrics_path}")

        # Plot predicted vs observed
        plot_predicted_vs_observed(
            y_test, y_pred_xgb,
            str(RUSH_MODELS_DIR / f"{TASK_NAME}_xgboost_pred_vs_obs.png"),
            title="XGBoost: Predicted vs Observed ICU LOS"
        )
        plot_predicted_vs_observed(
            y_test, y_pred_en,
            str(RUSH_MODELS_DIR / f"{TASK_NAME}_elasticnet_pred_vs_obs.png"),
            title="ElasticNet: Predicted vs Observed ICU LOS"
        )
        print("Predicted vs Observed plots saved.")

    else:
        print("\n" + "="*60)
        print(f"NON-RUSH SITE ({SITE_NAME}): Running experiments")
        print("="*60)
        print("Experiments will be run in the next cell...")
    return (all_results, ci_en, ci_xgb, en_model, evaluator, metrics_en, metrics_xgb,
            xgb_model, y_pred_en, y_pred_xgb)


@app.cell
def _(
    ElasticNetModel, IS_RUSH, RESULTS_DIR, RUSH_MODELS_DIR, SITE_NAME,
    TASK_NAME, TASK_TYPE, TaskEvaluator, X_test, X_train, XGBoostModel,
    all_results, datetime, json, np, plot_predicted_vs_observed, print_metrics, y_test, y_train
):
    if not IS_RUSH:
        # ============================================================
        # NON-RUSH SITE: Run 3 experiments
        # ============================================================

        # Create output directory
        _output_dir = RESULTS_DIR / SITE_NAME / TASK_NAME
        _output_dir.mkdir(parents=True, exist_ok=True)

        _evaluator = TaskEvaluator(TASK_TYPE)
        _experiments_results = {}

        # ----------------------------------------------------------
        # EXPERIMENT 1: Rush Model Evaluation
        # ----------------------------------------------------------
        print("\n" + "="*60)
        print("EXPERIMENT 1: Rush Model Evaluation")
        print("="*60)

        _rush_xgb_path = RUSH_MODELS_DIR / f"{TASK_NAME}_xgboost.json"
        _rush_en_path = RUSH_MODELS_DIR / f"{TASK_NAME}_elasticnet.joblib"

        if _rush_xgb_path.exists() and _rush_en_path.exists():
            # Load Rush models
            _rush_xgb = XGBoostModel(task_type=TASK_TYPE).load(str(_rush_xgb_path))
            _rush_xgb.feature_names = list(X_test.columns)
            _rush_en = ElasticNetModel(task_type=TASK_TYPE).load(str(_rush_en_path))

            # Evaluate on local test data
            _y_pred_rush_xgb = _rush_xgb.predict(X_test)
            _metrics_rush_xgb = _evaluator.evaluate(y_test, _y_pred_rush_xgb)
            _ci_rush_xgb = _evaluator.compute_bootstrap_ci(y_test, _y_pred_rush_xgb, n_iterations=1000)
            print_metrics(_metrics_rush_xgb, "Rush XGBoost on Local Test")

            _y_pred_rush_en = _rush_en.predict(X_test)
            _metrics_rush_en = _evaluator.evaluate(y_test, _y_pred_rush_en)
            _ci_rush_en = _evaluator.compute_bootstrap_ci(y_test, _y_pred_rush_en, n_iterations=1000)
            print_metrics(_metrics_rush_en, "Rush ElasticNet on Local Test")

            _experiments_results['rush_eval'] = {
                "site": SITE_NAME,
                "task": TASK_NAME,
                "experiment": "rush_eval",
                "timestamp": datetime.now().isoformat(),
                "n_test": len(y_test),
                "models": {
                    "xgboost": _ci_rush_xgb,
                    "elasticnet": _ci_rush_en
                }
            }

            # Save results
            with open(_output_dir / "rush_eval_metrics.json", 'w') as _f:
                json.dump(_experiments_results['rush_eval'], _f, indent=2)
            print(f"Saved: {_output_dir / 'rush_eval_metrics.json'}")

            # Plot predicted vs observed
            plot_predicted_vs_observed(
                y_test, _y_pred_rush_xgb,
                str(_output_dir / "rush_xgboost_pred_vs_obs.png"),
                title="Rush XGBoost: Predicted vs Observed ICU LOS"
            )
            plot_predicted_vs_observed(
                y_test, _y_pred_rush_en,
                str(_output_dir / "rush_elasticnet_pred_vs_obs.png"),
                title="Rush ElasticNet: Predicted vs Observed ICU LOS"
            )

        else:
            print(f"WARNING: Rush models not found at {RUSH_MODELS_DIR}")
            print("  Skipping Rush evaluation experiment")

        # ----------------------------------------------------------
        # EXPERIMENT 2: Transfer Learning
        # ----------------------------------------------------------
        print("\n" + "="*60)
        print("EXPERIMENT 2: Transfer Learning")
        print("="*60)

        if _rush_xgb_path.exists() and _rush_en_path.exists():
            # Load Rush models for fine-tuning
            _transfer_xgb = XGBoostModel(task_type=TASK_TYPE).load(str(_rush_xgb_path))
            _transfer_xgb.fine_tune(X_train, y_train, lr_multiplier=0.1, num_rounds=100, verbose=True)

            # Evaluate
            _y_pred_transfer_xgb = _transfer_xgb.predict(X_test)
            _metrics_transfer_xgb = _evaluator.evaluate(y_test, _y_pred_transfer_xgb)
            _ci_transfer_xgb = _evaluator.compute_bootstrap_ci(y_test, _y_pred_transfer_xgb, n_iterations=1000)
            print_metrics(_metrics_transfer_xgb, "Transfer XGBoost on Local Test")

            # Save transfer model
            _transfer_xgb.save(str(_output_dir / "transfer_xgboost.json"))

            # ElasticNet transfer: Use Rush preprocessing, retrain on local data
            _rush_en_loaded = ElasticNetModel(task_type=TASK_TYPE).load(str(_rush_en_path))
            _transfer_en = ElasticNetModel(task_type=TASK_TYPE)
            _transfer_en.feature_medians = _rush_en_loaded.feature_medians  # Use Rush medians
            _transfer_en.train(X_train, y_train, verbose=True)

            _y_pred_transfer_en = _transfer_en.predict(X_test)
            _metrics_transfer_en = _evaluator.evaluate(y_test, _y_pred_transfer_en)
            _ci_transfer_en = _evaluator.compute_bootstrap_ci(y_test, _y_pred_transfer_en, n_iterations=1000)
            print_metrics(_metrics_transfer_en, "Transfer ElasticNet on Local Test")

            # Save transfer model
            _transfer_en.save(str(_output_dir / "transfer_elasticnet.joblib"))

            _experiments_results['transfer_learning'] = {
                "site": SITE_NAME,
                "task": TASK_NAME,
                "experiment": "transfer_learning",
                "timestamp": datetime.now().isoformat(),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "models": {
                    "xgboost": _ci_transfer_xgb,
                    "elasticnet": _ci_transfer_en
                }
            }

            with open(_output_dir / "transfer_learning_metrics.json", 'w') as _f:
                json.dump(_experiments_results['transfer_learning'], _f, indent=2)
            print(f"Saved: {_output_dir / 'transfer_learning_metrics.json'}")

            # Plot predicted vs observed
            plot_predicted_vs_observed(
                y_test, _y_pred_transfer_xgb,
                str(_output_dir / "transfer_xgboost_pred_vs_obs.png"),
                title="Transfer XGBoost: Predicted vs Observed ICU LOS"
            )
            plot_predicted_vs_observed(
                y_test, _y_pred_transfer_en,
                str(_output_dir / "transfer_elasticnet_pred_vs_obs.png"),
                title="Transfer ElasticNet: Predicted vs Observed ICU LOS"
            )

        else:
            print("Skipping transfer learning (Rush models not found)")

        # ----------------------------------------------------------
        # EXPERIMENT 3: Independent Training
        # ----------------------------------------------------------
        print("\n" + "="*60)
        print("EXPERIMENT 3: Independent Training")
        print("="*60)

        # Train XGBoost from scratch
        _indep_xgb = XGBoostModel(task_type=TASK_TYPE)
        _indep_xgb.train(X_train, y_train, verbose=True)

        _y_pred_indep_xgb = _indep_xgb.predict(X_test)
        _metrics_indep_xgb = _evaluator.evaluate(y_test, _y_pred_indep_xgb)
        _ci_indep_xgb = _evaluator.compute_bootstrap_ci(y_test, _y_pred_indep_xgb, n_iterations=1000)
        print_metrics(_metrics_indep_xgb, "Independent XGBoost on Local Test")

        _indep_xgb.save(str(_output_dir / "independent_xgboost.json"))

        # Train ElasticNet from scratch
        _indep_en = ElasticNetModel(task_type=TASK_TYPE)
        _indep_en.train(X_train, y_train, verbose=True)

        _y_pred_indep_en = _indep_en.predict(X_test)
        _metrics_indep_en = _evaluator.evaluate(y_test, _y_pred_indep_en)
        _ci_indep_en = _evaluator.compute_bootstrap_ci(y_test, _y_pred_indep_en, n_iterations=1000)
        print_metrics(_metrics_indep_en, "Independent ElasticNet on Local Test")

        _indep_en.save(str(_output_dir / "independent_elasticnet.joblib"))

        _experiments_results['independent'] = {
            "site": SITE_NAME,
            "task": TASK_NAME,
            "experiment": "independent",
            "timestamp": datetime.now().isoformat(),
            "n_train": len(y_train),
            "n_test": len(y_test),
            "models": {
                "xgboost": _ci_indep_xgb,
                "elasticnet": _ci_indep_en
            }
        }

        with open(_output_dir / "independent_metrics.json", 'w') as _f:
            json.dump(_experiments_results['independent'], _f, indent=2)
        print(f"Saved: {_output_dir / 'independent_metrics.json'}")

        # Plot predicted vs observed
        plot_predicted_vs_observed(
            y_test, _y_pred_indep_xgb,
            str(_output_dir / "independent_xgboost_pred_vs_obs.png"),
            title="Independent XGBoost: Predicted vs Observed ICU LOS"
        )
        plot_predicted_vs_observed(
            y_test, _y_pred_indep_en,
            str(_output_dir / "independent_elasticnet_pred_vs_obs.png"),
            title="Independent ElasticNet: Predicted vs Observed ICU LOS"
        )

        # Update all_results
        all_results.update(_experiments_results)

        print("\n" + "="*60)
        print("All experiments complete!")
        print(f"Results saved to: {_output_dir}")
        print("="*60)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Results Summary""")
    return


@app.cell
def _(IS_RUSH, SITE_NAME, TASK_NAME, all_results, pd):
    # Create summary table
    if all_results:
        _rows = []
        for _exp_name, _exp_data in all_results.items():
            if 'models' in _exp_data:
                for _model_name, _model_metrics in _exp_data['models'].items():
                    _row = {
                        'experiment': _exp_name,
                        'model': _model_name
                    }
                    for _metric, _values in _model_metrics.items():
                        if isinstance(_values, dict) and 'mean' in _values:
                            _row[_metric] = f"{_values['mean']:.3f} [{_values['ci_lower']:.3f}-{_values['ci_upper']:.3f}]"
                        else:
                            _row[_metric] = _values
                    _rows.append(_row)

        summary_df = pd.DataFrame(_rows)
        print(f"\n=== Results Summary for {TASK_NAME} ({SITE_NAME}) ===")
        print(summary_df.to_string(index=False))
    else:
        summary_df = None
        print("No results to display")
    return (summary_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Next Steps

    - Run `03b_task6_mortality.py` for Task 6 (Hospital Mortality)
    - Run `03c_task7_readmission.py` for Task 7 (ICU Readmission)
    """
    )
    return


if __name__ == "__main__":
    app.run()
