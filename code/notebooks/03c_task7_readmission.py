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
    # Task 7: ICU Readmission Prediction

    **Task Type:** Binary Classification
    **Label:** `label_icu_readmission`
    **Metrics:** AUROC, AUPRC, F1, Accuracy, Precision, Recall

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
    compute_ici = _eval_module.compute_ici
    compute_dca = _eval_module.compute_dca
    compute_roc_curve_data = _eval_module.compute_roc_curve_data
    plot_roc_curve = _eval_module.plot_roc_curve
    plot_dca_curve = _eval_module.plot_dca_curve
    plot_calibration_curve = _eval_module.plot_calibration_curve

    print("=== Task 7: ICU Readmission ===")
    return (
        ElasticNetModel, Path, PROJECT_ROOT, TaskEvaluator, XGBoostModel,
        compute_dca, compute_ici, compute_roc_curve_data,
        datetime, importlib, json, np, os, pd, plot_calibration_curve, plot_dca_curve, plot_roc_curve,
        print_metrics, sys
    )


@app.cell
def _(PROJECT_ROOT, Path, json):
    # Configuration
    CONFIG_PATH = PROJECT_ROOT / "clif_config.json"
    FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
    RUSH_MODELS_DIR = PROJECT_ROOT / "rush_models"
    RESULTS_DIR = PROJECT_ROOT / "results_to_box"

    # Task configuration
    TASK_NAME = "task7_icu_readmission"
    TASK_TYPE = "classification"
    LABEL_COL = "label_icu_readmission"

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

    # Check label distribution
    if LABEL_COL in features_df.columns:
        _label_counts = features_df[LABEL_COL].value_counts()
        print(f"  Label distribution:")
        print(f"    Class 0: {_label_counts.get(0, 0)} ({_label_counts.get(0, 0)/len(features_df)*100:.1f}%)")
        print(f"    Class 1: {_label_counts.get(1, 0)} ({_label_counts.get(1, 0)/len(features_df)*100:.1f}%)")
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
    y_train = _train_df[LABEL_COL].values.astype(int)
    X_test = _test_df[_feature_cols]
    y_test = _test_df[LABEL_COL].values.astype(int)

    print(f"Data prepared:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  Features: {len(_feature_cols)}")
    print(f"  Train class balance: {y_train.sum()}/{len(y_train)} ({y_train.mean()*100:.1f}% positive)")
    print(f"  Test class balance: {y_test.sum()}/{len(y_test)} ({y_test.mean()*100:.1f}% positive)")
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
    TaskEvaluator, X_test, X_train, XGBoostModel,
    compute_dca, compute_ici, compute_roc_curve_data,
    datetime, json, np, plot_calibration_curve, plot_dca_curve, plot_roc_curve,
    print_metrics, y_test, y_train
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
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)
        y_pred_class_xgb = (y_pred_proba_xgb >= 0.5).astype(int)
        metrics_xgb = evaluator.evaluate(y_test, y_pred_class_xgb, y_pred_proba_xgb)
        ci_xgb = evaluator.compute_bootstrap_ci(y_test, y_pred_class_xgb, y_pred_proba_xgb, n_iterations=1000)
        print_metrics(metrics_xgb, "XGBoost Test Metrics")

        # Compute ICI, ROC, DCA for XGBoost
        _ici_xgb = compute_ici(y_test, y_pred_proba_xgb)
        _roc_xgb = compute_roc_curve_data(y_test, y_pred_proba_xgb)
        _dca_xgb = compute_dca(y_test, y_pred_proba_xgb)
        ci_xgb['ici'] = {'mean': _ici_xgb['ici'], 'ci_lower': None, 'ci_upper': None}
        ci_xgb['roc_curve'] = _roc_xgb
        ci_xgb['dca_curve'] = _dca_xgb
        ci_xgb['calibration_curve'] = _ici_xgb['calibration_curve']
        print(f"  ICI (XGBoost): {_ici_xgb['ici']:.4f}")

        # Plot ROC, DCA, and Calibration for XGBoost
        plot_roc_curve(_roc_xgb, str(RUSH_MODELS_DIR / f"{TASK_NAME}_xgboost_roc.png"),
                       title=f"{TASK_NAME} - XGBoost ROC Curve")
        plot_dca_curve(_dca_xgb, str(RUSH_MODELS_DIR / f"{TASK_NAME}_xgboost_dca.png"),
                       title=f"{TASK_NAME} - XGBoost Decision Curve")
        plot_calibration_curve(y_test, y_pred_proba_xgb,
                               str(RUSH_MODELS_DIR / f"{TASK_NAME}_xgboost_calibration.png"),
                               title=f"{TASK_NAME} - XGBoost Calibration")

        # Save model
        xgb_model.save(str(RUSH_MODELS_DIR / f"{TASK_NAME}_xgboost.json"))

        # --- Train ElasticNet ---
        print("\n--- Training ElasticNet ---")
        en_model = ElasticNetModel(task_type=TASK_TYPE)
        en_model.train(X_train, y_train, verbose=True)

        # Evaluate
        y_pred_proba_en = en_model.predict_proba(X_test)
        y_pred_class_en = (y_pred_proba_en >= 0.5).astype(int)
        metrics_en = evaluator.evaluate(y_test, y_pred_class_en, y_pred_proba_en)
        ci_en = evaluator.compute_bootstrap_ci(y_test, y_pred_class_en, y_pred_proba_en, n_iterations=1000)
        print_metrics(metrics_en, "ElasticNet Test Metrics")

        # Compute ICI, ROC, DCA for ElasticNet
        _ici_en = compute_ici(y_test, y_pred_proba_en)
        _roc_en = compute_roc_curve_data(y_test, y_pred_proba_en)
        _dca_en = compute_dca(y_test, y_pred_proba_en)
        ci_en['ici'] = {'mean': _ici_en['ici'], 'ci_lower': None, 'ci_upper': None}
        ci_en['roc_curve'] = _roc_en
        ci_en['dca_curve'] = _dca_en
        ci_en['calibration_curve'] = _ici_en['calibration_curve']
        print(f"  ICI (ElasticNet): {_ici_en['ici']:.4f}")

        # Plot ROC, DCA, and Calibration for ElasticNet
        plot_roc_curve(_roc_en, str(RUSH_MODELS_DIR / f"{TASK_NAME}_elasticnet_roc.png"),
                       title=f"{TASK_NAME} - ElasticNet ROC Curve")
        plot_dca_curve(_dca_en, str(RUSH_MODELS_DIR / f"{TASK_NAME}_elasticnet_dca.png"),
                       title=f"{TASK_NAME} - ElasticNet Decision Curve")
        plot_calibration_curve(y_test, y_pred_proba_en,
                               str(RUSH_MODELS_DIR / f"{TASK_NAME}_elasticnet_calibration.png"),
                               title=f"{TASK_NAME} - ElasticNet Calibration")

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

    else:
        print("\n" + "="*60)
        print(f"NON-RUSH SITE ({SITE_NAME}): Running experiments")
        print("="*60)
        print("Experiments will be run in the next cell...")
    return (all_results, ci_en, ci_xgb, en_model, evaluator, metrics_en, metrics_xgb,
            xgb_model, y_pred_class_en, y_pred_class_xgb, y_pred_proba_en,
            y_pred_proba_xgb)


@app.cell
def _(
    ElasticNetModel, IS_RUSH, RESULTS_DIR, RUSH_MODELS_DIR, SITE_NAME,
    TASK_NAME, TASK_TYPE, TaskEvaluator, X_test, X_train, XGBoostModel,
    all_results, compute_dca, compute_ici, compute_roc_curve_data,
    datetime, json, np, plot_calibration_curve, plot_dca_curve, plot_roc_curve, print_metrics,
    y_test, y_train
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
            _y_pred_proba_rush_xgb = _rush_xgb.predict_proba(X_test)
            _y_pred_class_rush_xgb = (_y_pred_proba_rush_xgb >= 0.5).astype(int)
            _metrics_rush_xgb = _evaluator.evaluate(y_test, _y_pred_class_rush_xgb, _y_pred_proba_rush_xgb)
            _ci_rush_xgb = _evaluator.compute_bootstrap_ci(y_test, _y_pred_class_rush_xgb, _y_pred_proba_rush_xgb, n_iterations=1000)
            print_metrics(_metrics_rush_xgb, "Rush XGBoost on Local Test")

            _y_pred_proba_rush_en = _rush_en.predict_proba(X_test)
            _y_pred_class_rush_en = (_y_pred_proba_rush_en >= 0.5).astype(int)
            _metrics_rush_en = _evaluator.evaluate(y_test, _y_pred_class_rush_en, _y_pred_proba_rush_en)
            _ci_rush_en = _evaluator.compute_bootstrap_ci(y_test, _y_pred_class_rush_en, _y_pred_proba_rush_en, n_iterations=1000)
            print_metrics(_metrics_rush_en, "Rush ElasticNet on Local Test")

            # Compute ICI, ROC, DCA for Rush models
            _ici_rush_xgb = compute_ici(y_test, _y_pred_proba_rush_xgb)
            _roc_rush_xgb = compute_roc_curve_data(y_test, _y_pred_proba_rush_xgb)
            _dca_rush_xgb = compute_dca(y_test, _y_pred_proba_rush_xgb)
            _ci_rush_xgb['ici'] = {'mean': _ici_rush_xgb['ici'], 'ci_lower': None, 'ci_upper': None}
            _ci_rush_xgb['roc_curve'] = _roc_rush_xgb
            _ci_rush_xgb['dca_curve'] = _dca_rush_xgb
            _ci_rush_xgb['calibration_curve'] = _ici_rush_xgb['calibration_curve']

            _ici_rush_en = compute_ici(y_test, _y_pred_proba_rush_en)
            _roc_rush_en = compute_roc_curve_data(y_test, _y_pred_proba_rush_en)
            _dca_rush_en = compute_dca(y_test, _y_pred_proba_rush_en)
            _ci_rush_en['ici'] = {'mean': _ici_rush_en['ici'], 'ci_lower': None, 'ci_upper': None}
            _ci_rush_en['roc_curve'] = _roc_rush_en
            _ci_rush_en['dca_curve'] = _dca_rush_en
            _ci_rush_en['calibration_curve'] = _ici_rush_en['calibration_curve']

            print(f"  ICI (Rush XGBoost): {_ici_rush_xgb['ici']:.4f}")
            print(f"  ICI (Rush ElasticNet): {_ici_rush_en['ici']:.4f}")

            # Create experiment subfolder
            _rush_eval_dir = _output_dir / "01_rush_eval"
            _rush_eval_dir.mkdir(parents=True, exist_ok=True)

            # Plot ROC, DCA, and Calibration
            plot_roc_curve(_roc_rush_xgb, str(_rush_eval_dir / "xgboost_roc.png"),
                           title=f"{TASK_NAME} - Rush XGBoost ROC")
            plot_dca_curve(_dca_rush_xgb, str(_rush_eval_dir / "xgboost_dca.png"),
                           title=f"{TASK_NAME} - Rush XGBoost DCA")
            plot_calibration_curve(y_test, _y_pred_proba_rush_xgb,
                                   str(_rush_eval_dir / "xgboost_calibration.png"),
                                   title=f"{TASK_NAME} - Rush XGBoost Calibration")
            plot_roc_curve(_roc_rush_en, str(_rush_eval_dir / "elasticnet_roc.png"),
                           title=f"{TASK_NAME} - Rush ElasticNet ROC")
            plot_dca_curve(_dca_rush_en, str(_rush_eval_dir / "elasticnet_dca.png"),
                           title=f"{TASK_NAME} - Rush ElasticNet DCA")
            plot_calibration_curve(y_test, _y_pred_proba_rush_en,
                                   str(_rush_eval_dir / "elasticnet_calibration.png"),
                                   title=f"{TASK_NAME} - Rush ElasticNet Calibration")

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
            with open(_rush_eval_dir / "metrics.json", 'w') as _f:
                json.dump(_experiments_results['rush_eval'], _f, indent=2)
            print(f"Saved: {_rush_eval_dir / 'metrics.json'}")

            # ----------------------------------------------------------
            # EXPERIMENT 1b: Rush Model + Platt Scaling
            # ----------------------------------------------------------
            print("\n" + "="*60)
            print("EXPERIMENT 1b: Rush Model + Platt Scaling")
            print("="*60)

            from sklearn.calibration import CalibratedClassifierCV

            # Wrapper to use Rush model with CalibratedClassifierCV
            class _PretrainedClassifier:
                """Wrapper for pre-trained model to work with CalibratedClassifierCV"""
                _estimator_type = "classifier"

                def __init__(self, model):
                    self._model = model
                    self.classes_ = np.array([0, 1])

                def fit(self, X, y):
                    """Dummy fit method - model is already trained"""
                    return self

                def predict(self, X):
                    proba = self._model.predict_proba(X)
                    return (proba >= 0.5).astype(int)

                def predict_proba(self, X):
                    proba = self._model.predict_proba(X)
                    return np.column_stack([1 - proba, proba])

            # Platt scaling for XGBoost
            _rush_xgb_wrapper = _PretrainedClassifier(_rush_xgb)
            _calibrator_xgb = CalibratedClassifierCV(_rush_xgb_wrapper, cv="prefit", method="sigmoid")
            _calibrator_xgb.fit(X_train, y_train)

            _y_pred_proba_platt_xgb = _calibrator_xgb.predict_proba(X_test)[:, 1]
            _y_pred_class_platt_xgb = (_y_pred_proba_platt_xgb >= 0.5).astype(int)
            _metrics_platt_xgb = _evaluator.evaluate(y_test, _y_pred_class_platt_xgb, _y_pred_proba_platt_xgb)
            _ci_platt_xgb = _evaluator.compute_bootstrap_ci(y_test, _y_pred_class_platt_xgb, _y_pred_proba_platt_xgb, n_iterations=1000)
            print_metrics(_metrics_platt_xgb, "Rush XGBoost + Platt Scaling on Local Test")

            # Platt scaling for ElasticNet
            _rush_en_wrapper = _PretrainedClassifier(_rush_en)
            _calibrator_en = CalibratedClassifierCV(_rush_en_wrapper, cv="prefit", method="sigmoid")
            _calibrator_en.fit(X_train, y_train)

            _y_pred_proba_platt_en = _calibrator_en.predict_proba(X_test)[:, 1]
            _y_pred_class_platt_en = (_y_pred_proba_platt_en >= 0.5).astype(int)
            _metrics_platt_en = _evaluator.evaluate(y_test, _y_pred_class_platt_en, _y_pred_proba_platt_en)
            _ci_platt_en = _evaluator.compute_bootstrap_ci(y_test, _y_pred_class_platt_en, _y_pred_proba_platt_en, n_iterations=1000)
            print_metrics(_metrics_platt_en, "Rush ElasticNet + Platt Scaling on Local Test")

            # Compute ICI, ROC, DCA for Platt-scaled models
            _ici_platt_xgb = compute_ici(y_test, _y_pred_proba_platt_xgb)
            _roc_platt_xgb = compute_roc_curve_data(y_test, _y_pred_proba_platt_xgb)
            _dca_platt_xgb = compute_dca(y_test, _y_pred_proba_platt_xgb)
            _ci_platt_xgb['ici'] = {'mean': _ici_platt_xgb['ici'], 'ci_lower': None, 'ci_upper': None}
            _ci_platt_xgb['roc_curve'] = _roc_platt_xgb
            _ci_platt_xgb['dca_curve'] = _dca_platt_xgb
            _ci_platt_xgb['calibration_curve'] = _ici_platt_xgb['calibration_curve']

            _ici_platt_en = compute_ici(y_test, _y_pred_proba_platt_en)
            _roc_platt_en = compute_roc_curve_data(y_test, _y_pred_proba_platt_en)
            _dca_platt_en = compute_dca(y_test, _y_pred_proba_platt_en)
            _ci_platt_en['ici'] = {'mean': _ici_platt_en['ici'], 'ci_lower': None, 'ci_upper': None}
            _ci_platt_en['roc_curve'] = _roc_platt_en
            _ci_platt_en['dca_curve'] = _dca_platt_en
            _ci_platt_en['calibration_curve'] = _ici_platt_en['calibration_curve']

            print(f"  ICI (Platt XGBoost): {_ici_platt_xgb['ici']:.4f}")
            print(f"  ICI (Platt ElasticNet): {_ici_platt_en['ici']:.4f}")

            # Create experiment subfolder
            _platt_dir = _output_dir / "02_platt_scaling"
            _platt_dir.mkdir(parents=True, exist_ok=True)

            # Plot ROC, DCA, and Calibration for Platt-scaled models
            plot_roc_curve(_roc_platt_xgb, str(_platt_dir / "xgboost_roc.png"),
                           title=f"{TASK_NAME} - Platt XGBoost ROC")
            plot_dca_curve(_dca_platt_xgb, str(_platt_dir / "xgboost_dca.png"),
                           title=f"{TASK_NAME} - Platt XGBoost DCA")
            plot_calibration_curve(y_test, _y_pred_proba_platt_xgb,
                                   str(_platt_dir / "xgboost_calibration.png"),
                                   title=f"{TASK_NAME} - Platt XGBoost Calibration")
            plot_roc_curve(_roc_platt_en, str(_platt_dir / "elasticnet_roc.png"),
                           title=f"{TASK_NAME} - Platt ElasticNet ROC")
            plot_dca_curve(_dca_platt_en, str(_platt_dir / "elasticnet_dca.png"),
                           title=f"{TASK_NAME} - Platt ElasticNet DCA")
            plot_calibration_curve(y_test, _y_pred_proba_platt_en,
                                   str(_platt_dir / "elasticnet_calibration.png"),
                                   title=f"{TASK_NAME} - Platt ElasticNet Calibration")

            _experiments_results['platt_scaling'] = {
                "site": SITE_NAME,
                "task": TASK_NAME,
                "experiment": "platt_scaling",
                "timestamp": datetime.now().isoformat(),
                "n_train": len(y_train),
                "n_test": len(y_test),
                "models": {
                    "xgboost": _ci_platt_xgb,
                    "elasticnet": _ci_platt_en
                }
            }

            with open(_platt_dir / "metrics.json", 'w') as _f:
                json.dump(_experiments_results['platt_scaling'], _f, indent=2)
            print(f"Saved: {_platt_dir / 'metrics.json'}")

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
            # Create experiment subfolder
            _transfer_dir = _output_dir / "03_transfer_learning"
            _transfer_dir.mkdir(parents=True, exist_ok=True)

            # Load Rush models for fine-tuning
            _transfer_xgb = XGBoostModel(task_type=TASK_TYPE).load(str(_rush_xgb_path))
            _transfer_xgb.fine_tune(X_train, y_train, lr_multiplier=0.1, num_rounds=100, verbose=True)

            # Evaluate
            _y_pred_proba_transfer_xgb = _transfer_xgb.predict_proba(X_test)
            _y_pred_class_transfer_xgb = (_y_pred_proba_transfer_xgb >= 0.5).astype(int)
            _metrics_transfer_xgb = _evaluator.evaluate(y_test, _y_pred_class_transfer_xgb, _y_pred_proba_transfer_xgb)
            _ci_transfer_xgb = _evaluator.compute_bootstrap_ci(y_test, _y_pred_class_transfer_xgb, _y_pred_proba_transfer_xgb, n_iterations=1000)
            print_metrics(_metrics_transfer_xgb, "Transfer XGBoost on Local Test")

            # Save transfer model
            _transfer_xgb.save(str(_transfer_dir / "xgboost.json"))

            # ElasticNet transfer: Use Rush preprocessing, retrain on local data
            _rush_en_loaded = ElasticNetModel(task_type=TASK_TYPE).load(str(_rush_en_path))
            _transfer_en = ElasticNetModel(task_type=TASK_TYPE)
            _transfer_en.feature_medians = _rush_en_loaded.feature_medians  # Use Rush medians
            _transfer_en.train(X_train, y_train, verbose=True)

            _y_pred_proba_transfer_en = _transfer_en.predict_proba(X_test)
            _y_pred_class_transfer_en = (_y_pred_proba_transfer_en >= 0.5).astype(int)
            _metrics_transfer_en = _evaluator.evaluate(y_test, _y_pred_class_transfer_en, _y_pred_proba_transfer_en)
            _ci_transfer_en = _evaluator.compute_bootstrap_ci(y_test, _y_pred_class_transfer_en, _y_pred_proba_transfer_en, n_iterations=1000)
            print_metrics(_metrics_transfer_en, "Transfer ElasticNet on Local Test")

            # Save transfer model
            _transfer_en.save(str(_transfer_dir / "elasticnet.joblib"))

            # Compute ICI, ROC, DCA for Transfer models
            _ici_transfer_xgb = compute_ici(y_test, _y_pred_proba_transfer_xgb)
            _roc_transfer_xgb = compute_roc_curve_data(y_test, _y_pred_proba_transfer_xgb)
            _dca_transfer_xgb = compute_dca(y_test, _y_pred_proba_transfer_xgb)
            _ci_transfer_xgb['ici'] = {'mean': _ici_transfer_xgb['ici'], 'ci_lower': None, 'ci_upper': None}
            _ci_transfer_xgb['roc_curve'] = _roc_transfer_xgb
            _ci_transfer_xgb['dca_curve'] = _dca_transfer_xgb
            _ci_transfer_xgb['calibration_curve'] = _ici_transfer_xgb['calibration_curve']

            _ici_transfer_en = compute_ici(y_test, _y_pred_proba_transfer_en)
            _roc_transfer_en = compute_roc_curve_data(y_test, _y_pred_proba_transfer_en)
            _dca_transfer_en = compute_dca(y_test, _y_pred_proba_transfer_en)
            _ci_transfer_en['ici'] = {'mean': _ici_transfer_en['ici'], 'ci_lower': None, 'ci_upper': None}
            _ci_transfer_en['roc_curve'] = _roc_transfer_en
            _ci_transfer_en['dca_curve'] = _dca_transfer_en
            _ci_transfer_en['calibration_curve'] = _ici_transfer_en['calibration_curve']

            print(f"  ICI (Transfer XGBoost): {_ici_transfer_xgb['ici']:.4f}")
            print(f"  ICI (Transfer ElasticNet): {_ici_transfer_en['ici']:.4f}")

            # Plot ROC, DCA, and Calibration
            plot_roc_curve(_roc_transfer_xgb, str(_transfer_dir / "xgboost_roc.png"),
                           title=f"{TASK_NAME} - Transfer XGBoost ROC")
            plot_dca_curve(_dca_transfer_xgb, str(_transfer_dir / "xgboost_dca.png"),
                           title=f"{TASK_NAME} - Transfer XGBoost DCA")
            plot_calibration_curve(y_test, _y_pred_proba_transfer_xgb,
                                   str(_transfer_dir / "xgboost_calibration.png"),
                                   title=f"{TASK_NAME} - Transfer XGBoost Calibration")
            plot_roc_curve(_roc_transfer_en, str(_transfer_dir / "elasticnet_roc.png"),
                           title=f"{TASK_NAME} - Transfer ElasticNet ROC")
            plot_dca_curve(_dca_transfer_en, str(_transfer_dir / "elasticnet_dca.png"),
                           title=f"{TASK_NAME} - Transfer ElasticNet DCA")
            plot_calibration_curve(y_test, _y_pred_proba_transfer_en,
                                   str(_transfer_dir / "elasticnet_calibration.png"),
                                   title=f"{TASK_NAME} - Transfer ElasticNet Calibration")

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

            with open(_transfer_dir / "metrics.json", 'w') as _f:
                json.dump(_experiments_results['transfer_learning'], _f, indent=2)
            print(f"Saved: {_transfer_dir / 'metrics.json'}")

        else:
            print("Skipping transfer learning (Rush models not found)")

        # ----------------------------------------------------------
        # EXPERIMENT 3: Independent Training
        # ----------------------------------------------------------
        print("\n" + "="*60)
        print("EXPERIMENT 3: Independent Training")
        print("="*60)

        # Create experiment subfolder
        _indep_dir = _output_dir / "04_independent"
        _indep_dir.mkdir(parents=True, exist_ok=True)

        # Train XGBoost from scratch
        _indep_xgb = XGBoostModel(task_type=TASK_TYPE)
        _indep_xgb.train(X_train, y_train, verbose=True)

        _y_pred_proba_indep_xgb = _indep_xgb.predict_proba(X_test)
        _y_pred_class_indep_xgb = (_y_pred_proba_indep_xgb >= 0.5).astype(int)
        _metrics_indep_xgb = _evaluator.evaluate(y_test, _y_pred_class_indep_xgb, _y_pred_proba_indep_xgb)
        _ci_indep_xgb = _evaluator.compute_bootstrap_ci(y_test, _y_pred_class_indep_xgb, _y_pred_proba_indep_xgb, n_iterations=1000)
        print_metrics(_metrics_indep_xgb, "Independent XGBoost on Local Test")

        _indep_xgb.save(str(_indep_dir / "xgboost.json"))

        # Train ElasticNet from scratch
        _indep_en = ElasticNetModel(task_type=TASK_TYPE)
        _indep_en.train(X_train, y_train, verbose=True)

        _y_pred_proba_indep_en = _indep_en.predict_proba(X_test)
        _y_pred_class_indep_en = (_y_pred_proba_indep_en >= 0.5).astype(int)
        _metrics_indep_en = _evaluator.evaluate(y_test, _y_pred_class_indep_en, _y_pred_proba_indep_en)
        _ci_indep_en = _evaluator.compute_bootstrap_ci(y_test, _y_pred_class_indep_en, _y_pred_proba_indep_en, n_iterations=1000)
        print_metrics(_metrics_indep_en, "Independent ElasticNet on Local Test")

        _indep_en.save(str(_indep_dir / "elasticnet.joblib"))

        # Compute ICI, ROC, DCA for Independent models
        _ici_indep_xgb = compute_ici(y_test, _y_pred_proba_indep_xgb)
        _roc_indep_xgb = compute_roc_curve_data(y_test, _y_pred_proba_indep_xgb)
        _dca_indep_xgb = compute_dca(y_test, _y_pred_proba_indep_xgb)
        _ci_indep_xgb['ici'] = {'mean': _ici_indep_xgb['ici'], 'ci_lower': None, 'ci_upper': None}
        _ci_indep_xgb['roc_curve'] = _roc_indep_xgb
        _ci_indep_xgb['dca_curve'] = _dca_indep_xgb
        _ci_indep_xgb['calibration_curve'] = _ici_indep_xgb['calibration_curve']

        _ici_indep_en = compute_ici(y_test, _y_pred_proba_indep_en)
        _roc_indep_en = compute_roc_curve_data(y_test, _y_pred_proba_indep_en)
        _dca_indep_en = compute_dca(y_test, _y_pred_proba_indep_en)
        _ci_indep_en['ici'] = {'mean': _ici_indep_en['ici'], 'ci_lower': None, 'ci_upper': None}
        _ci_indep_en['roc_curve'] = _roc_indep_en
        _ci_indep_en['dca_curve'] = _dca_indep_en
        _ci_indep_en['calibration_curve'] = _ici_indep_en['calibration_curve']

        print(f"  ICI (Independent XGBoost): {_ici_indep_xgb['ici']:.4f}")
        print(f"  ICI (Independent ElasticNet): {_ici_indep_en['ici']:.4f}")

        # Plot ROC, DCA, and Calibration
        plot_roc_curve(_roc_indep_xgb, str(_indep_dir / "xgboost_roc.png"),
                       title=f"{TASK_NAME} - Independent XGBoost ROC")
        plot_dca_curve(_dca_indep_xgb, str(_indep_dir / "xgboost_dca.png"),
                       title=f"{TASK_NAME} - Independent XGBoost DCA")
        plot_calibration_curve(y_test, _y_pred_proba_indep_xgb,
                               str(_indep_dir / "xgboost_calibration.png"),
                               title=f"{TASK_NAME} - Independent XGBoost Calibration")
        plot_roc_curve(_roc_indep_en, str(_indep_dir / "elasticnet_roc.png"),
                       title=f"{TASK_NAME} - Independent ElasticNet ROC")
        plot_dca_curve(_dca_indep_en, str(_indep_dir / "elasticnet_dca.png"),
                       title=f"{TASK_NAME} - Independent ElasticNet DCA")
        plot_calibration_curve(y_test, _y_pred_proba_indep_en,
                               str(_indep_dir / "elasticnet_calibration.png"),
                               title=f"{TASK_NAME} - Independent ElasticNet Calibration")

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

        with open(_indep_dir / "metrics.json", 'w') as _f:
            json.dump(_experiments_results['independent'], _f, indent=2)
        print(f"Saved: {_indep_dir / 'metrics.json'}")

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
                        # Skip curve data (lists/dicts)
                        if _metric in ['roc_curve', 'dca_curve', 'calibration_curve']:
                            continue
                        if isinstance(_values, dict) and 'mean' in _values:
                            if _values.get('ci_lower') is not None and _values.get('ci_upper') is not None:
                                _row[_metric] = f"{_values['mean']:.3f} [{_values['ci_lower']:.3f}-{_values['ci_upper']:.3f}]"
                            else:
                                _row[_metric] = f"{_values['mean']:.3f}"
                        elif not isinstance(_values, (dict, list)):
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
    ## All Tasks Complete!

    All 3 FLAIR benchmark tasks have been processed:
    - Task 5: ICU Length of Stay (Regression)
    - Task 6: Hospital Mortality (Classification)
    - Task 7: ICU Readmission (Classification)
    """
    )
    return


if __name__ == "__main__":
    app.run()
