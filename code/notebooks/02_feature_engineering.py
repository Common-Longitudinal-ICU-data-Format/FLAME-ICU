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
    sys.path.append('../..')

    import pandas as pd
    import polars as pl
    from pathlib import Path
    import warnings
    warnings.filterwarnings('ignore')

    from code.features import FeatureExtractor, extract_features_for_task

    print("=== FLAIR Feature Engineering ===")
    return FeatureExtractor, Path, extract_features_for_task, pd, pl, sys, warnings


@app.cell
def _(Path):
    # Configuration
    CONFIG_PATH = "../../clif_config.json"
    DATASETS_DIR = Path("../../outputs/datasets")
    FEATURES_DIR = Path("../../outputs/features")
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    TASKS = [
        "task5_icu_los",
        "task6_hospital_mortality",
        "task7_icu_readmission"
    ]

    print(f"Configuration:")
    print(f"  CLIF config: {CONFIG_PATH}")
    print(f"  Datasets dir: {DATASETS_DIR}")
    print(f"  Features output dir: {FEATURES_DIR}")
    return CONFIG_PATH, DATASETS_DIR, FEATURES_DIR, TASKS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Task Datasets""")
    return


@app.cell
def _(DATASETS_DIR, TASKS, pl):
    # Load task datasets
    task_datasets = {}

    for task_name in TASKS:
        dataset_path = DATASETS_DIR / f"{task_name}.parquet"
        if dataset_path.exists():
            df = pl.read_parquet(dataset_path)
            task_datasets[task_name] = df
            print(f"Loaded {task_name}: {df.shape}")
        else:
            print(f"WARNING: {task_name} dataset not found at {dataset_path}")
            print("  Please run 01_task_generator.py first")

    print(f"\nLoaded {len(task_datasets)} task datasets")
    return dataset_path, df, task_datasets, task_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Extract Features for Each Task""")
    return


@app.cell
def _(CONFIG_PATH, FEATURES_DIR, extract_features_for_task, pd, task_datasets):
    # Extract features for each task
    task_features = {}

    for task_name, task_df in task_datasets.items():
        print(f"\n{'='*60}")
        print(f"Extracting features for {task_name}...")
        print(f"{'='*60}")

        try:
            # Convert to pandas for feature extraction
            task_pandas = task_df.to_pandas()

            # Extract features
            features_df = extract_features_for_task(
                config_path=CONFIG_PATH,
                task_dataset=task_pandas,
                time_col_start='window_start',
                time_col_end='window_end'
            )

            task_features[task_name] = features_df

            # Save features
            output_path = FEATURES_DIR / f"{task_name}_features.parquet"
            features_df.to_parquet(output_path, index=False)

            print(f"  Features shape: {features_df.shape}")
            print(f"  Saved to: {output_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"{'='*60}")
    return features_df, output_path, task_features, task_name, task_pandas


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Merge Features with Labels""")
    return


@app.cell
def _(FEATURES_DIR, pd, task_datasets, task_features):
    # Merge features with labels and save final datasets
    final_datasets = {}

    # Define label columns for each task
    label_cols = {
        'task5_icu_los': 'icu_los_hours',
        'task6_hospital_mortality': 'label_mortality',
        'task7_icu_readmission': 'label_icu_readmission'
    }

    for task_name, features_df in task_features.items():
        print(f"\nMerging {task_name}...")

        # Get original task dataset with labels
        task_df = task_datasets[task_name].to_pandas()
        label_col = label_cols.get(task_name)

        # Columns to keep from task dataset
        keep_cols = ['hospitalization_id', 'split']
        if label_col and label_col in task_df.columns:
            keep_cols.append(label_col)

        # Merge features with labels
        labels_df = task_df[keep_cols].drop_duplicates()
        merged_df = pd.merge(features_df, labels_df, on='hospitalization_id', how='inner')

        final_datasets[task_name] = merged_df

        # Save final dataset
        output_path = FEATURES_DIR / f"{task_name}_final.parquet"
        merged_df.to_parquet(output_path, index=False)

        print(f"  Final shape: {merged_df.shape}")
        print(f"  Train: {len(merged_df[merged_df['split'] == 'train'])}")
        print(f"  Test: {len(merged_df[merged_df['split'] == 'test'])}")
        print(f"  Saved to: {output_path}")

    return final_datasets, keep_cols, label_col, label_cols, labels_df, merged_df, output_path, task_df, task_name


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Feature Summary""")
    return


@app.cell
def _(final_datasets, pd):
    # Display feature summary
    for task_name, df in final_datasets.items():
        print(f"\n=== {task_name} ===")
        print(f"Shape: {df.shape}")

        # Feature columns (exclude ID, split, label)
        exclude = ['hospitalization_id', 'split', 'icu_los_hours', 'label_mortality', 'label_icu_readmission']
        feature_cols = [c for c in df.columns if c not in exclude]
        print(f"Feature columns: {len(feature_cols)}")

        # Missing value summary
        missing = df[feature_cols].isnull().sum()
        missing_pct = (missing / len(df) * 100).round(1)
        high_missing = missing_pct[missing_pct > 50]
        if len(high_missing) > 0:
            print(f"High missing (>50%): {list(high_missing.index)}")

        print(f"\nFeature list:")
        for i, col in enumerate(feature_cols[:20]):
            print(f"  {i+1}. {col}")
        if len(feature_cols) > 20:
            print(f"  ... and {len(feature_cols) - 20} more")

    return col, df, exclude, feature_cols, high_missing, i, missing, missing_pct, task_name


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
