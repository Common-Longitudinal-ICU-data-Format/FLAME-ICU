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
    # FLAIR Feature Engineering

    This notebook extracts features from CLIF tables for tasks 5, 6, and 7.

    ## Prerequisites
    1. Run `01_task_generator.py` to generate task datasets
    2. Task datasets should be in `outputs/datasets/`

    ## Feature Sources
    - **Vitals**: Heart rate, MAP, SBP, respiratory rate, SpO2, temperature
    - **Labs**: Albumin, creatinine, lactate, platelets, etc.
    - **Respiratory Support**: Device type, FiO2, PEEP
    - **Medications**: Vasopressor usage
    - **Patient Assessments**: GCS total

    ## Aggregation Strategy
    - **Max/Worst**: lactate, BUN, creatinine, AST, ALT, INR, etc.
    - **Min/Worst**: platelets, PaO2, SpO2, SBP, albumin
    - **Median**: respiratory rate, FiO2
    - **Last**: GCS total
    - **Derived**: vasopressor count, device one-hot encoding
    """
    )
    return


@app.cell
def _():
    import sys
    import os
    from pathlib import Path

    # Insert project root at beginning of path to override built-in 'code' module
    _project_root = str(Path(os.getcwd()).resolve())
    # If running from notebooks dir, go up two levels
    if _project_root.endswith('notebooks'):
        _project_root = str(Path(_project_root).parent.parent)
    elif _project_root.endswith('code'):
        _project_root = str(Path(_project_root).parent)

    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    import pandas as pd
    import polars as pl
    import warnings
    warnings.filterwarnings('ignore')

    # Import using importlib to avoid name conflict with built-in 'code' module
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "features",
        Path(_project_root) / "code" / "features.py"
    )
    _features_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_features_module)
    FeatureExtractor = _features_module.FeatureExtractor
    extract_features_for_task = _features_module.extract_features_for_task

    print("=== FLAIR Feature Engineering ===")
    return FeatureExtractor, Path, extract_features_for_task, pd, pl, sys, warnings


@app.cell
def _(Path):
    import os as _os

    # Determine project root (works whether running from project root or notebooks dir)
    _cwd = Path(_os.getcwd()).resolve()
    if _cwd.name == 'notebooks':
        PROJECT_ROOT = _cwd.parent.parent
    elif _cwd.name == 'code':
        PROJECT_ROOT = _cwd.parent
    else:
        PROJECT_ROOT = _cwd

    # Configuration using absolute paths
    CONFIG_PATH = str(PROJECT_ROOT / "clif_config.json")
    DATASETS_DIR = PROJECT_ROOT / "outputs" / "datasets"
    FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    TASKS = [
        "task5_icu_los",
        "task6_hospital_mortality",
        "task7_icu_readmission"
    ]

    print(f"Configuration:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  CLIF config: {CONFIG_PATH}")
    print(f"  Datasets dir: {DATASETS_DIR}")
    print(f"  Features output dir: {FEATURES_DIR}")
    return CONFIG_PATH, DATASETS_DIR, FEATURES_DIR, PROJECT_ROOT, TASKS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Task Datasets""")
    return


@app.cell
def _(DATASETS_DIR, TASKS, pl):
    # Load task datasets
    task_datasets = {}

    for _task_name in TASKS:
        _dataset_path = DATASETS_DIR / f"{_task_name}.parquet"
        if _dataset_path.exists():
            _df = pl.read_parquet(_dataset_path)
            task_datasets[_task_name] = _df
            print(f"Loaded {_task_name}: {_df.shape}")
        else:
            print(f"WARNING: {_task_name} dataset not found at {_dataset_path}")
            print("  Please run 01_task_generator.py first")

    print(f"\nLoaded {len(task_datasets)} task datasets")
    return (task_datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Extract Features for Each Task""")
    return


@app.cell
def _(CONFIG_PATH, FEATURES_DIR, extract_features_for_task, pd, task_datasets):
    # Extract features for each task
    task_features = {}

    for _task_name, _task_df in task_datasets.items():
        print(f"\n{'='*60}")
        print(f"Extracting features for {_task_name}...")
        print(f"{'='*60}")

        try:
            # Convert to pandas for feature extraction
            _task_pandas = _task_df.to_pandas()

            # Extract features
            _features_df = extract_features_for_task(
                config_path=CONFIG_PATH,
                task_dataset=_task_pandas,
                time_col_start='window_start',
                time_col_end='window_end'
            )

            task_features[_task_name] = _features_df
            print(f"  Features shape: {_features_df.shape}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"{'='*60}")
    return (task_features,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Merge Features with Labels""")
    return


@app.cell
def _(FEATURES_DIR, pd, task_datasets, task_features):
    # Merge features with labels and save final datasets
    final_datasets = {}

    # Define label columns for each task
    _label_cols = {
        'task5_icu_los': 'icu_los_hours',
        'task6_hospital_mortality': 'label_mortality',
        'task7_icu_readmission': 'label_icu_readmission'
    }

    for _task_name, _features_df in task_features.items():
        print(f"\nMerging {_task_name}...")

        # Get original task dataset with labels
        _task_df = task_datasets[_task_name].to_pandas()
        _label_col = _label_cols.get(_task_name)

        # Columns to keep from task dataset
        _keep_cols = ['hospitalization_id', 'split']
        if _label_col and _label_col in _task_df.columns:
            _keep_cols.append(_label_col)

        # Merge features with labels
        _labels_df = _task_df[_keep_cols].drop_duplicates()
        _merged_df = pd.merge(_features_df, _labels_df, on='hospitalization_id', how='inner')

        final_datasets[_task_name] = _merged_df

        # Save final dataset
        _output_path = FEATURES_DIR / f"{_task_name}_final.parquet"
        _merged_df.to_parquet(_output_path, index=False)

        print(f"  Final shape: {_merged_df.shape}")
        print(f"  Train: {len(_merged_df[_merged_df['split'] == 'train'])}")
        print(f"  Test: {len(_merged_df[_merged_df['split'] == 'test'])}")
        print(f"  Saved to: {_output_path}")

    return (final_datasets,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Feature Summary""")
    return


@app.cell
def _(final_datasets, pd):
    # Display feature summary
    for _task_name, _df in final_datasets.items():
        print(f"\n=== {_task_name} ===")
        print(f"Shape: {_df.shape}")

        # Feature columns (exclude ID, split, label)
        _exclude = ['hospitalization_id', 'split', 'icu_los_hours', 'label_mortality', 'label_icu_readmission']
        _feature_cols = [c for c in _df.columns if c not in _exclude]
        print(f"Feature columns: {len(_feature_cols)}")

        # Missing value summary
        _missing = _df[_feature_cols].isnull().sum()
        _missing_pct = (_missing / len(_df) * 100).round(1)
        _high_missing = _missing_pct[_missing_pct > 50]
        if len(_high_missing) > 0:
            print(f"High missing (>50%): {list(_high_missing.index)}")

        print(f"\nFeature list:")
        for _i, _col in enumerate(_feature_cols[:20]):
            print(f"  {_i+1}. {_col}")
        if len(_feature_cols) > 20:
            print(f"  ... and {len(_feature_cols) - 20} more")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Next Steps

    Run `03_train_evaluate.py` to train XGBoost and ElasticNet models.
    """
    )
    return


if __name__ == "__main__":
    app.run()
