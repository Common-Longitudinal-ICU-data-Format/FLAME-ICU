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
    # Table 1: Clinical Cohort Summary Statistics

    This notebook generates Table 1 summary statistics for FLAIR benchmark tasks 5, 6, and 7.

    ## Prerequisites
    - Run `02_feature_engineering.py` to generate feature datasets

    ## Output
    - JSON files: `results_to_box/table1_task{N}_{site}.json`
    - CSV files: `results_to_box/table1_task{N}_{site}.csv`

    ## Statistics Generated
    - **Continuous variables**: N, Missing%, Mean, SD, Median, IQR (Q25-Q75)
    - **Categorical variables**: N total, Missing%, Count/Percentage per unique value (dynamically discovered)
    - **Label distribution**: Separate statistics for train/test splits
    """
    )
    return


@app.cell
def _():
    import sys
    import os
    import json
    from pathlib import Path

    import polars as pl
    import numpy as np

    print("=== Table 1 Generator ===")
    return Path, json, np, os, pl, sys


@app.cell
def _(Path, json, os):
    # Determine project root
    _cwd = Path(os.getcwd()).resolve()
    if _cwd.name == 'notebooks':
        PROJECT_ROOT = _cwd.parent.parent
    elif _cwd.name == 'code':
        PROJECT_ROOT = _cwd.parent
    else:
        PROJECT_ROOT = _cwd

    # Load site name from config
    _config_path = PROJECT_ROOT / "clif_config.json"
    with open(_config_path) as _f:
        _config = json.load(_f)
    SITE_NAME = _config.get("site", "unknown")

    # Directories
    FEATURES_DIR = PROJECT_ROOT / "outputs" / "features"
    RESULTS_DIR = PROJECT_ROOT / "results_to_box"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Task configuration
    TASKS = {
        "task5_icu_los": {
            "file": "task5_icu_los_final.parquet",
            "label_col": "icu_los_hours",
            "task_type": "regression",
            "description": "ICU Length of Stay (hours)"
        },
        "task6_hospital_mortality": {
            "file": "task6_hospital_mortality_final.parquet",
            "label_col": "label_mortality",
            "task_type": "classification",
            "description": "Hospital Mortality"
        },
        "task7_icu_readmission": {
            "file": "task7_icu_readmission_final.parquet",
            "label_col": "label_icu_readmission",
            "task_type": "classification",
            "description": "ICU Readmission"
        }
    }

    print(f"Configuration:")
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Site name: {SITE_NAME}")
    print(f"  Features dir: {FEATURES_DIR}")
    print(f"  Results dir: {RESULTS_DIR}")
    return FEATURES_DIR, PROJECT_ROOT, RESULTS_DIR, SITE_NAME, TASKS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Helper Functions""")
    return


@app.cell
def _(pl):
    def get_continuous_stats(series: pl.Series, var_name: str) -> dict:
        """
        Calculate statistics for continuous variables using Polars.
        Returns: n, n_missing, missing_pct, mean, std, median, q25, q75, min, max
        """
        n_total = len(series)
        n_missing = series.null_count()
        n_valid = n_total - n_missing

        stats = {
            "variable": var_name,
            "n": int(n_valid),
            "n_missing": int(n_missing),
            "missing_pct": round(float(n_missing / n_total * 100), 2) if n_total > 0 else 0.0,
        }

        if n_valid > 0:
            s = series.cast(pl.Float64)
            stats.update({
                "mean": round(float(s.mean()), 2),
                "std": round(float(s.std()), 2),
                "median": round(float(s.median()), 2),
                "q25": round(float(s.quantile(0.25)), 2),
                "q75": round(float(s.quantile(0.75)), 2),
                "min": round(float(s.min()), 2),
                "max": round(float(s.max()), 2),
            })
        else:
            stats.update({
                "mean": None, "std": None, "median": None,
                "q25": None, "q75": None, "min": None, "max": None
            })

        return stats

    return (get_continuous_stats,)


@app.cell
def _(pl):
    def get_categorical_stats(series: pl.Series, var_name: str) -> dict:
        """
        Calculate statistics for categorical variables using Polars.
        DYNAMICALLY discovers all unique values - no hardcoding.
        """
        n_total = len(series)
        n_missing = series.null_count()

        # Get value counts - DYNAMICALLY finds all unique values
        value_counts = series.value_counts().sort("count", descending=True)

        # Build categories dict dynamically
        categories = {}
        for row in value_counts.iter_rows(named=True):
            cat_value = row.get(series.name) or row.get("value")
            count = row["count"]

            if cat_value is not None:  # Exclude null from categories
                categories[str(cat_value)] = {
                    "count": int(count),
                    "percentage": round(float(count / n_total * 100), 2)
                }

        return {
            "variable": var_name,
            "n_total": int(n_total),
            "n_missing": int(n_missing),
            "missing_pct": round(float(n_missing / n_total * 100), 2) if n_total > 0 else 0.0,
            "categories": categories
        }

    return (get_categorical_stats,)


@app.cell
def _(pl):
    def get_binary_stats(series: pl.Series, var_name: str) -> dict:
        """
        Calculate statistics for binary (0/1) variables.
        Returns count and percentage of positive cases (value=1).
        """
        n_total = len(series)
        n_missing = series.null_count()
        n_valid = n_total - n_missing

        if n_valid > 0:
            n_positive = int(series.sum())
            pct_positive = round(float(n_positive / n_total * 100), 2)
        else:
            n_positive = 0
            pct_positive = 0.0

        return {
            "variable": var_name,
            "n_total": int(n_total),
            "n_missing": int(n_missing),
            "missing_pct": round(float(n_missing / n_total * 100), 2) if n_total > 0 else 0.0,
            "n_positive": n_positive,
            "pct_positive": pct_positive
        }

    return (get_binary_stats,)


@app.cell
def _(get_binary_stats, get_continuous_stats, pl):
    def get_label_stats_by_split(df: pl.DataFrame, label_col: str, task_type: str) -> dict:
        """
        Compute label statistics separately for train and test splits.
        """
        result = {}

        for split_name in ["train", "test"]:
            split_df = df.filter(pl.col("split") == split_name)

            if len(split_df) > 0:
                if task_type == "regression":
                    result[split_name] = get_continuous_stats(split_df[label_col], f"{label_col}_{split_name}")
                else:
                    result[split_name] = get_binary_stats(split_df[label_col], f"{label_col}_{split_name}")
            else:
                result[split_name] = {"n": 0, "n_missing": 0, "missing_pct": 0.0}

        return result

    return (get_label_stats_by_split,)


@app.cell
def _():
    def classify_columns(df_columns: list, task_config: dict) -> dict:
        """
        Dynamically classify columns into groups for Table 1.
        """
        label_col = task_config["label_col"]

        # Exclude columns
        exclude = ['hospitalization_id', 'split', label_col]

        # Known categorical columns (demographics)
        categorical = ['sex_category', 'race_category', 'ethnicity_category']

        # Binary device columns (detected dynamically)
        device_cols = [c for c in df_columns if c.startswith('device_')]

        # Binary and count columns
        binary_cols = ['isfemale']
        count_cols = ['vasopressor_count']

        # Age columns - treat as continuous
        age_cols = ['age_at_admission', 'age']

        # Everything else is continuous clinical features
        continuous = []
        for c in df_columns:
            if c not in exclude + categorical + device_cols + binary_cols + count_cols + age_cols:
                continuous.append(c)

        return {
            "exclude": exclude,
            "categorical": [c for c in categorical if c in df_columns],
            "device_binary": device_cols,
            "binary": [c for c in binary_cols if c in df_columns],
            "count": [c for c in count_cols if c in df_columns],
            "age": [c for c in age_cols if c in df_columns],
            "continuous": continuous,
            "label": label_col
        }

    return (classify_columns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Generate Table 1""")
    return


@app.cell
def _(classify_columns, get_binary_stats, get_categorical_stats, get_continuous_stats, get_label_stats_by_split, pl):
    def generate_table1(df: pl.DataFrame, task_name: str, task_config: dict, site_name: str) -> dict:
        """
        Generate complete Table 1 statistics for a task dataset.
        """
        col_groups = classify_columns(df.columns, task_config)

        table1 = {
            "site": site_name,
            "task": task_name,
            "task_description": task_config["description"],
            "task_type": task_config["task_type"],
            "cohort_info": {
                "n_hospitalizations": len(df),
                "n_train": len(df.filter(pl.col("split") == "train")),
                "n_test": len(df.filter(pl.col("split") == "test"))
            },
            "demographics": {
                "categorical": {},
                "continuous": {}
            },
            "clinical_features": {
                "vitals": {},
                "labs": {},
                "respiratory": {},
                "assessments": {}
            },
            "respiratory_devices": {},
            "medications": {},
            "outcomes": {}
        }

        # === Demographics - Categorical (DYNAMIC) ===
        for col in col_groups["categorical"]:
            table1["demographics"]["categorical"][col] = get_categorical_stats(df[col], col)

        # === Demographics - Binary (isfemale) ===
        for col in col_groups["binary"]:
            table1["demographics"]["categorical"][col] = get_binary_stats(df[col], col)

        # === Demographics - Continuous (age) ===
        for col in col_groups["age"]:
            table1["demographics"]["continuous"][col] = get_continuous_stats(df[col], col)

        # === Clinical Features - Group by Type ===
        vital_prefixes = ['heart_rate', 'map', 'sbp', 'respiratory_rate', 'spo2', 'temp_c']
        lab_prefixes = ['albumin', 'alt', 'ast', 'bicarbonate', 'bilirubin', 'bun', 'chloride',
                       'creatinine', 'inr', 'lactate', 'platelet', 'po2', 'potassium',
                       'pt', 'ptt', 'sodium', 'wbc']
        resp_prefixes = ['fio2', 'peep']

        for col in col_groups["continuous"]:
            if any(col.startswith(v) for v in vital_prefixes):
                table1["clinical_features"]["vitals"][col] = get_continuous_stats(df[col], col)
            elif any(col.startswith(lab) for lab in lab_prefixes):
                table1["clinical_features"]["labs"][col] = get_continuous_stats(df[col], col)
            elif any(col.startswith(r) for r in resp_prefixes):
                table1["clinical_features"]["respiratory"][col] = get_continuous_stats(df[col], col)
            elif col.startswith('gcs'):
                table1["clinical_features"]["assessments"][col] = get_continuous_stats(df[col], col)

        # === Respiratory Devices (Binary) ===
        for col in col_groups["device_binary"]:
            table1["respiratory_devices"][col] = get_binary_stats(df[col], col)

        # === Medications (vasopressor_count as categorical distribution) ===
        for col in col_groups["count"]:
            # Cast to string to get categorical distribution (0, 1, 2, 3, etc.)
            table1["medications"][col] = get_categorical_stats(df[col].cast(pl.Utf8), col)

        # === Outcomes (by split) ===
        label_col = task_config["label_col"]
        if label_col in df.columns:
            table1["outcomes"][label_col] = get_label_stats_by_split(
                df, label_col, task_config["task_type"]
            )

        return table1

    return (generate_table1,)


@app.cell
def _():
    def flatten_table1_to_csv(table1: dict) -> list:
        """
        Flatten nested Table 1 statistics into rows for CSV export.
        Returns list of dicts with consistent columns.
        """
        rows = []

        # Helper to add continuous stats
        def add_continuous(var_name, stats, section, category=""):
            rows.append({
                "Section": section,
                "Variable": var_name,
                "Category": category,
                "N": stats.get("n"),
                "N_Missing": stats.get("n_missing"),
                "Missing_Pct": stats.get("missing_pct"),
                "Mean": stats.get("mean"),
                "SD": stats.get("std"),
                "Median": stats.get("median"),
                "Q25": stats.get("q25"),
                "Q75": stats.get("q75"),
                "Count": None,
                "Percentage": None
            })

        # Helper to add categorical stats
        def add_categorical(var_name, stats, section):
            # Add row for each category dynamically discovered
            for cat_name, cat_stats in stats.get("categories", {}).items():
                rows.append({
                    "Section": section,
                    "Variable": var_name,
                    "Category": cat_name,
                    "N": stats.get("n_total"),
                    "N_Missing": stats.get("n_missing"),
                    "Missing_Pct": stats.get("missing_pct"),
                    "Mean": None,
                    "SD": None,
                    "Median": None,
                    "Q25": None,
                    "Q75": None,
                    "Count": cat_stats.get("count"),
                    "Percentage": cat_stats.get("percentage")
                })

        # Helper to add binary stats
        def add_binary(var_name, stats, section):
            rows.append({
                "Section": section,
                "Variable": var_name,
                "Category": "Yes",
                "N": stats.get("n_total"),
                "N_Missing": stats.get("n_missing"),
                "Missing_Pct": stats.get("missing_pct"),
                "Mean": None,
                "SD": None,
                "Median": None,
                "Q25": None,
                "Q75": None,
                "Count": stats.get("n_positive"),
                "Percentage": stats.get("pct_positive")
            })

        # Cohort info
        cohort = table1["cohort_info"]
        rows.append({
            "Section": "Cohort",
            "Variable": "N_hospitalizations",
            "Category": "Total",
            "N": cohort["n_hospitalizations"],
            "N_Missing": 0,
            "Missing_Pct": 0.0,
            "Mean": None, "SD": None, "Median": None, "Q25": None, "Q75": None,
            "Count": cohort["n_hospitalizations"],
            "Percentage": 100.0
        })
        rows.append({
            "Section": "Cohort",
            "Variable": "N_hospitalizations",
            "Category": "Train",
            "N": cohort["n_train"],
            "N_Missing": 0,
            "Missing_Pct": 0.0,
            "Mean": None, "SD": None, "Median": None, "Q25": None, "Q75": None,
            "Count": cohort["n_train"],
            "Percentage": round(cohort["n_train"] / cohort["n_hospitalizations"] * 100, 2)
        })
        rows.append({
            "Section": "Cohort",
            "Variable": "N_hospitalizations",
            "Category": "Test",
            "N": cohort["n_test"],
            "N_Missing": 0,
            "Missing_Pct": 0.0,
            "Mean": None, "SD": None, "Median": None, "Q25": None, "Q75": None,
            "Count": cohort["n_test"],
            "Percentage": round(cohort["n_test"] / cohort["n_hospitalizations"] * 100, 2)
        })

        # Demographics - Categorical
        for var_name, stats in table1["demographics"]["categorical"].items():
            if "categories" in stats:
                add_categorical(var_name, stats, "Demographics")
            elif "n_positive" in stats:
                add_binary(var_name, stats, "Demographics")

        # Demographics - Continuous
        for var_name, stats in table1["demographics"]["continuous"].items():
            add_continuous(var_name, stats, "Demographics")

        # Clinical Features
        for subsection, features in table1["clinical_features"].items():
            for var_name, stats in features.items():
                add_continuous(var_name, stats, f"Clinical_{subsection.title()}")

        # Respiratory Devices
        for var_name, stats in table1["respiratory_devices"].items():
            add_binary(var_name, stats, "Respiratory_Devices")

        # Medications (vasopressor_count as categorical distribution)
        for var_name, stats in table1["medications"].items():
            add_categorical(var_name, stats, "Medications")

        # Outcomes (by split)
        for var_name, split_stats in table1["outcomes"].items():
            for split_name, stats in split_stats.items():
                if "mean" in stats:  # Continuous (regression)
                    add_continuous(f"{var_name} ({split_name})", stats, "Outcomes")
                elif "n_positive" in stats:  # Binary (classification)
                    add_binary(f"{var_name} ({split_name})", stats, "Outcomes")

        return rows

    return (flatten_table1_to_csv,)


@app.cell
def _(FEATURES_DIR, SITE_NAME, TASKS, generate_table1, pl):
    # Load datasets and generate Table 1 for each task
    all_tables = {}

    for _task_name, _config in TASKS.items():
        _file_path = FEATURES_DIR / _config["file"]

        if _file_path.exists():
            print(f"\n{'='*60}")
            print(f"Processing: {_task_name}")
            print(f"{'='*60}")

            _df = pl.read_parquet(_file_path)
            print(f"  Loaded: {_df.shape[0]} rows, {_df.shape[1]} columns")

            _table1 = generate_table1(_df, _task_name, _config, SITE_NAME)
            all_tables[_task_name] = _table1

            print(f"  Generated Table 1 with {_table1['cohort_info']['n_hospitalizations']} hospitalizations")
            print(f"  Site: {_table1['site']}")
        else:
            print(f"WARNING: {_file_path} not found")
            print("  Please run 02_feature_engineering.py first")

    print(f"\nGenerated Table 1 for {len(all_tables)} tasks")
    return (all_tables,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Save Outputs""")
    return


@app.cell
def _(RESULTS_DIR, SITE_NAME, all_tables, flatten_table1_to_csv, json, pl):
    # Save Table 1 to JSON and CSV files
    for _task_name, _table1 in all_tables.items():
        # JSON output (full nested structure)
        _json_path = RESULTS_DIR / f"table1_{_task_name}_{SITE_NAME}.json"
        with open(_json_path, 'w') as _fj:
            json.dump(_table1, _fj, indent=2)
        print(f"Saved JSON: {_json_path}")

        # CSV output (flattened)
        _csv_rows = flatten_table1_to_csv(_table1)
        _csv_df = pl.DataFrame(_csv_rows)
        _csv_path = RESULTS_DIR / f"table1_{_task_name}_{SITE_NAME}.csv"
        _csv_df.write_csv(_csv_path)
        print(f"Saved CSV: {_csv_path}")

    # Combined JSON file
    _combined_path = RESULTS_DIR / f"table1_all_tasks_{SITE_NAME}.json"
    with open(_combined_path, 'w') as _fc:
        json.dump(all_tables, _fc, indent=2)
    print(f"\nSaved combined JSON: {_combined_path}")

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Summary""")
    return


@app.cell
def _(SITE_NAME, all_tables, mo):
    # Summary comparison across tasks
    _summary_rows = []
    for _task_name, _table1 in all_tables.items():
        _cohort = _table1['cohort_info']
        _outcomes = _table1.get('outcomes', {})

        # Get outcome rate
        _outcome_str = ""
        for _label, _split_stats in _outcomes.items():
            if 'train' in _split_stats:
                _train_stats = _split_stats['train']
                _test_stats = _split_stats['test']
                if 'pct_positive' in _train_stats:
                    _outcome_str = f"Train: {_train_stats['pct_positive']}%, Test: {_test_stats['pct_positive']}%"
                elif 'mean' in _train_stats:
                    _outcome_str = f"Train: {_train_stats['mean']:.1f}h, Test: {_test_stats['mean']:.1f}h"

        _summary_rows.append(
            f"| {_task_name} | {_cohort['n_hospitalizations']:,} | {_cohort['n_train']:,} | {_cohort['n_test']:,} | {_outcome_str} |"
        )

    mo.md(f"""
## Summary for Site: {SITE_NAME}

| Task | N | Train | Test | Label Distribution |
|------|---|-------|------|-------------------|
{"".join(_summary_rows)}
    """)
    return


if __name__ == "__main__":
    app.run()
