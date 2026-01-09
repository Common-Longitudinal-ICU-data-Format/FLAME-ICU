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
    # FLAIR Task Dataset Generator

    This notebook generates datasets for FLAIR benchmark tasks 5, 6, and 7 using the FLAIR library.

    ## Tasks
    - **Task 5 (ICU LOS)**: Regression - predict ICU length of stay in hours using first 24 hours of data
    - **Task 6 (Hospital Mortality)**: Binary classification - predict in-hospital mortality using first 24 hours
    - **Task 7 (ICU Readmission)**: Binary classification - predict ICU readmission using entire first ICU stay

    ## Prerequisites
    1. Install FLAIR library: `cd FLAIR && uv pip install -e .`
    2. Create `clif_config.json` from template with your data paths
    3. Run this notebook to generate task datasets

    ## Output
    - `outputs/datasets/task5_icu_los.parquet`
    - `outputs/datasets/task6_hospital_mortality.parquet`
    - `outputs/datasets/task7_icu_readmission.parquet`
    """
    )
    return


@app.cell
def _():
    from flair import generate_task_dataset, TASK_REGISTRY, FLAIRCohortBuilder
    from pathlib import Path
    import polars as pl
    import json
    import os
    import warnings
    warnings.filterwarnings('ignore')

    print("=== FLAIR Task Dataset Generator ===")
    print(f"Available tasks: {list(TASK_REGISTRY.keys())}")
    return Path, generate_task_dataset, json, pl


@app.cell
def _(Path, json):
    # Configuration - use absolute paths based on project root
    PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
    CONFIG_PATH = str(PROJECT_ROOT / "clif_config.json")
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "datasets"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load config to detect site
    with open(CONFIG_PATH) as f:
        _config = json.load(f)
    SITE_NAME = _config.get("site", "").lower()
    IS_MIMIC = SITE_NAME == "mimic"

    # Temporal split configuration (only used for non-MIMIC sites)
    TRAIN_START = "2018-01-01"
    TRAIN_END = "2023-12-31"
    TEST_START = "2024-01-01"
    TEST_END = "2024-12-31"

    # Tasks to generate (5, 6, 7)
    TASKS = [
        "task5_icu_los",
        "task6_hospital_mortality",
        "task7_icu_readmission"
    ]

    print(f"Configuration:")
    print(f"  Config path: {CONFIG_PATH}")
    print(f"  Site: {SITE_NAME}")
    print(f"  Output directory: {OUTPUT_DIR}")
    if IS_MIMIC:
        print(f"  MIMIC detected - using built-in train/test splits (no date filtering)")
    else:
        print(f"  Train period: {TRAIN_START} to {TRAIN_END}")
        print(f"  Test period: {TEST_START} to {TEST_END}")
    print(f"  Tasks: {TASKS}")
    return (
        CONFIG_PATH,
        IS_MIMIC,
        OUTPUT_DIR,
        TASKS,
        TEST_END,
        TEST_START,
        TRAIN_END,
        TRAIN_START,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generate Task Datasets""")
    return


@app.cell
def _(
    CONFIG_PATH,
    IS_MIMIC,
    OUTPUT_DIR,
    TASKS,
    TEST_END,
    TEST_START,
    TRAIN_END,
    TRAIN_START,
    generate_task_dataset,
    pl,
):
    # Generate datasets for all tasks
    results = {}

    for _task_name in TASKS:
        print(f"\n{'='*60}")
        print(f"Generating {_task_name}...")
        print(f"{'='*60}")

        _output_path = str(OUTPUT_DIR / f"{_task_name}.parquet")

        try:
            # For MIMIC, no date parameters needed (uses built-in splits)
            # For other sites, use temporal split configuration
            if IS_MIMIC:
                _df = generate_task_dataset(
                    config_path=CONFIG_PATH,
                    task_name=_task_name,
                    output_path=_output_path,
                )
            else:
                _df = generate_task_dataset(
                    config_path=CONFIG_PATH,
                    task_name=_task_name,
                    train_start=TRAIN_START,
                    train_end=TRAIN_END,
                    test_start=TEST_START,
                    test_end=TEST_END,
                    output_path=_output_path,
                )

            results[_task_name] = _df

            # Print statistics
            _n_total = len(_df)
            _n_train = len(_df.filter(pl.col("split") == "train"))
            _n_test = len(_df.filter(pl.col("split") == "test"))

            print(f"  Total N: {_n_total}")
            print(f"  Train N: {_n_train}")
            print(f"  Test N: {_n_test}")
            print(f"  Saved to: {_output_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[_task_name] = None

    print(f"\n{'='*60}")
    print("Dataset generation complete!")
    print(f"{'='*60}")
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dataset Summary""")
    return


@app.cell
def _(results):
    results['task7_icu_readmission']
    return


@app.cell
def _(results):
    # Display summary statistics for each task
    for _task_name, _df in results.items():
        if _df is not None:
            print(f"\n=== {_task_name} ===")
            print(f"Shape: {_df.shape}")
            print(f"Columns: {_df.columns}")

            # Check for empty dataframe
            if len(_df) == 0:
                print("\n  WARNING: Dataset is empty! Check date ranges and data directory.")
                continue

            # Get label column (varies by task)
            if "icu_los_hours" in _df.columns:
                _label_col = "icu_los_hours"
                print(f"\nLabel statistics ({_label_col}):")
                print(f"  Mean: {_df[_label_col].mean():.2f} hours")
                print(f"  Median: {_df[_label_col].median():.2f} hours")
                print(f"  Min: {_df[_label_col].min():.2f} hours")
                print(f"  Max: {_df[_label_col].max():.2f} hours")
            elif "label_mortality" in _df.columns:
                _label_col = "label_mortality"
                _mortality_rate = _df[_label_col].mean()
                print(f"\nLabel statistics ({_label_col}):")
                print(f"  Mortality rate: {_mortality_rate:.3f} ({_mortality_rate*100:.1f}%)")
            elif "label_icu_readmission" in _df.columns:
                _label_col = "label_icu_readmission"
                _readmit_rate = _df[_label_col].mean()
                print(f"\nLabel statistics ({_label_col}):")
                print(f"  Readmission rate: {_readmit_rate:.3f} ({_readmit_rate*100:.1f}%)")

            # Split distribution
            print(f"\nSplit distribution:")
            _split_counts = _df.group_by("split").count()
            print(_split_counts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Next Steps

    1. Run `02_feature_engineering.py` to extract features using clifpy
    2. Run `03_train_evaluate.py` to train XGBoost and ElasticNet models
    """
    )
    return


if __name__ == "__main__":
    app.run()
