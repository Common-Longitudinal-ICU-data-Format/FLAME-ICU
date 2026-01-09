# FLAME-ICU Execution Guide

Complete step-by-step guide for running the FLAIR benchmark using FLAME-ICU.

------------------------------------------------------------------------

## Table of Contents

1.  [Prerequisites](#1-prerequisites)
2.  [Installation](#2-installation)
3.  [Configuration](#3-configuration)
4.  [Step 1: Generate Task Datasets](#step-1-generate-task-datasets)
5.  [Step 2: Feature Engineering](#step-2-feature-engineering)
6.  [Step 2b: Generate Table 1](#step-2b-generate-table-1-optional)
7.  [Step 3: Train & Evaluate Models](#step-3-train--evaluate-models)
8.  [Output Structure](#output-structure)
9.  [Uploading to BOX](#uploading-to-box)

------------------------------------------------------------------------

## 1. Prerequisites

### Install UV Package Manager

**Mac/Linux:**

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

``` powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Required CLIF Data

Ensure you have CLIF-formatted parquet files with these tables:

| Table | Required Columns |
|----------------------|--------------------------------------------------|
| hospitalization | All |
| patient | All |
| adt | All (location_category) |
| labs | hospitalization_id, lab_result_dttm, lab_category, lab_value |
| vitals | hospitalization_id, recorded_dttm, vital_category, vital_value |
| patient_assessments | All (gcs_total) |
| medication_admin_continuous | All (vasopressors) |
| respiratory_support | All (device_category, fio2_set, peep_set) |

------------------------------------------------------------------------

## 2. Installation

### Step 1: Clone the Repository

``` bash
git clone <repository-url>
cd FLAME-ICU
```

### Step 2: Clone and Install FLAIR Library

FLAIR is a separate repository that must be cloned into the FLAME-ICU directory.

> **Note:** FLAIR is currently in development. Soon it will be available via PyPI:
>
> ``` bash
> uv pip install flair-benchmark  # Coming soon!
> ```

**Current installation (clone from GitHub):**

``` bash
git clone https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR.git
cd FLAIR
uv pip install -e .
cd ..
```

### Step 3: Install Project Dependencies

``` bash
uv sync
```

### Verify Installation

``` bash
uv run python -c "from flair import generate_task_dataset; print('FLAIR OK')"
uv run python -c "from code.models import XGBoostModel; print('Models OK')"
```

------------------------------------------------------------------------

## 3. Configuration

### Create Configuration File

``` bash
cp clif_config_template.json clif_config.json
```

### Edit `clif_config.json`

``` json
{
    "site": "your_site_name",
    "data_directory": "/path/to/your/clif/parquet/files",
    "filetype": "parquet",
    "timezone": "US/Central"
}
```

**Site Names:** - `"rush"` - Rush University (trains baseline models) - `"mimic"` - MIMIC-IV (uses built-in splits) - `"site_a"`, `"site_b"`, etc. - Other consortium sites

**Timezone Examples:** - `"US/Central"` - Chicago - `"US/Eastern"` - New York - `"US/Pacific"` - Los Angeles

------------------------------------------------------------------------

## Step 1: Generate Task Datasets {#step-1-generate-task-datasets}

This step uses FLAIR's `generate_task_dataset()` function to create cohorts and labels.

### Run Notebook 01

**Interactive mode (recommended for first run):**

``` bash
uv run marimo edit code/notebooks/01_task_generator.py
```

### What FLAIR Does

``` python
from flair import generate_task_dataset

# Example: Generate Task 6 (Hospital Mortality)
df = generate_task_dataset(
    config_path="clif_config.json",
    task_name="task6_hospital_mortality",
    train_start="2018-01-01",  # Optional for MIMIC
    train_end="2023-12-31",
    test_start="2024-01-01",
    test_end="2024-12-31",
    output_path="outputs/datasets/task6.parquet"
)
```

### Output

```         
outputs/datasets/
├── task5_icu_los.parquet
├── task6_hospital_mortality.parquet
└── task7_icu_readmission.parquet
```

Each file contains: - `hospitalization_id` - Patient identifier - `split` - "train" or "test" - `window_start` - Feature collection start (ICU admission) - `window_end` - Prediction time - Task-specific label column

------------------------------------------------------------------------

## Step 2: Feature Engineering {#step-2-feature-engineering}

Extract clinical features from CLIF tables using clifpy.

### Run Notebook 02

``` bash
uv run marimo edit code/notebooks/02_feature_engineering.py
```

### Features Extracted

| Category | Features |
|------------------------------------|------------------------------------|
| Vitals | heart_rate, map, sbp, respiratory_rate, spo2, temp_c |
| Labs | albumin, alt, ast, bicarbonate, bilirubin, bun, chloride, creatinine, inr, lactate, platelet_count, po2_arterial, potassium, pt, ptt, sodium, wbc |
| Respiratory | device_category (one-hot), fio2_set, peep_set |
| Medications | vasopressor_count |
| Assessments | gcs_total |

### Output

```         
outputs/features/
├── task5_icu_los_final.parquet
├── task6_hospital_mortality_final.parquet
└── task7_icu_readmission_final.parquet
```

------------------------------------------------------------------------

## Step 2b: Generate Table 1 (Optional) {#step-2b-generate-table-1-optional}

Generate summary statistics tables for each task cohort.

### Run Notebook 02b

``` bash
uv run marimo edit code/notebooks/02b_table_one.py
```

Or headless:

``` bash
uv run python code/notebooks/02b_table_one.py
```

### What It Generates

-   **N hospitalizations** (total, train, test)
-   **Demographics**: age, sex, race, ethnicity (dynamically discovered from data)
-   **Clinical features**: vitals, labs, respiratory support, medications
-   **Missing percentages** for every variable
-   **Label distribution** by train/test split

### Output

```         
results_to_box/
├── table1_task5_icu_los_{site}.json
├── table1_task5_icu_los_{site}.csv
├── table1_task6_hospital_mortality_{site}.json
├── table1_task6_hospital_mortality_{site}.csv
├── table1_task7_icu_readmission_{site}.json
├── table1_task7_icu_readmission_{site}.csv
└── table1_all_tasks_{site}.json
```

------------------------------------------------------------------------

## Step 3: Train & Evaluate Models

The workflow differs based on your site.

### For Rush Site (`site: "rush"`)

Rush trains the baseline models that other sites will evaluate.

``` bash
# Task 5: ICU Length of Stay (Regression)
uv run marimo edit code/notebooks/03a_task5_icu_los.py

# Task 6: Hospital Mortality (Classification)
uv run marimo edit code/notebooks/03b_task6_mortality.py

# Task 7: ICU Readmission (Classification)
uv run marimo edit code/notebooks/03c_task7_readmission.py
```

**Output:**

```         
rush_models/
├── task5_icu_los_xgboost.json
├── task5_icu_los_elasticnet.joblib
├── task5_icu_los_metrics.json
├── task6_hospital_mortality_xgboost.json
├── task6_hospital_mortality_elasticnet.joblib
├── task6_hospital_mortality_metrics.json
├── task7_icu_readmission_xgboost.json
├── task7_icu_readmission_elasticnet.joblib
└── task7_icu_readmission_metrics.json
```

### For Other Sites (Non-Rush)

First, download Rush models from BOX and place in `rush_models/` folder.

Then run the same notebooks:

``` bash
uv run marimo edit code/notebooks/03a_task5_icu_los.py
uv run marimo edit code/notebooks/03b_task6_mortality.py
uv run marimo edit code/notebooks/03c_task7_readmission.py
```

**Four experiments run automatically:**

| Experiment | Description |
|-----------------------------------|-------------------------------------|
| `01_rush_eval` | Evaluate Rush models directly on your test data |
| `02_platt_scaling` | Rush models + Platt calibration using your train data |
| `03_transfer_learning` | Fine-tune Rush models with your train data |
| `04_independent` | Train models from scratch on your data |

**Output:**

```         
results_to_box/{your_site}/
├── task5_icu_los/
│   ├── 01_rush_eval/
│   │   ├── metrics.json
│   │   └── *.png (plots)
│   ├── 02_platt_scaling/
│   ├── 03_transfer_learning/
│   └── 04_independent/
├── task6_hospital_mortality/
│   └── ... (same structure)
└── task7_icu_readmission/
    └── ... (same structure)
```

------------------------------------------------------------------------

## Output Structure {#output-structure}

### Complete Directory Layout

```         
FLAME-ICU/
├── clif_config.json          # Your site configuration
├── outputs/
│   ├── datasets/             # Task cohorts from Step 1
│   └── features/             # Feature matrices from Step 2
├── rush_models/              # Rush baseline models (Step 3 - Rush only)
└── results_to_box/           # Experiment results (Step 3 - Non-Rush)
    └── {site_name}/
        └── {task_name}/
            ├── 01_rush_eval/
            ├── 02_platt_scaling/
            ├── 03_transfer_learning/
            └── 04_independent/
```

### Metrics JSON Structure

``` json
{
  "site": "your_site",
  "task": "task6_hospital_mortality",
  "experiment": "01_rush_eval",
  "n_train": 5000,
  "n_test": 2000,
  "models": {
    "xgboost": {
      "auroc": {"mean": 0.85, "ci_lower": 0.83, "ci_upper": 0.87},
      "auprc": {"mean": 0.45, "ci_lower": 0.42, "ci_upper": 0.48},
      "ici": {"mean": 0.05, "ci_lower": null, "ci_upper": null}
    },
    "elasticnet": { ... }
  }
}
```

------------------------------------------------------------------------

## Uploading to BOX {#uploading-to-box}

### Rush Site

Upload the `rush_models/` folder to:

```         
CLIF BOX / FLAME / rush_models/
```

### Other Sites

Upload the `results_to_box/{your_site}/` folder to:

```         
CLIF BOX / FLAME / {your_site}/
```

------------------------------------------------------------------------

## Quick Reference

### All Commands

``` bash
# Installation
git clone https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR.git
cd FLAIR && uv pip install -e . && cd ..
uv sync

# Configuration
cp clif_config_template.json clif_config.json
# Edit clif_config.json with your site info

# Step 1: Generate datasets
uv run marimo edit code/notebooks/01_task_generator.py

# Step 2: Feature engineering
uv run marimo edit code/notebooks/02_feature_engineering.py

# Step 2b: Generate Table 1 (optional)
uv run python code/notebooks/02b_table_one.py

# Step 3: Train/evaluate (run all 3)
uv run marimo edit code/notebooks/03a_task5_icu_los.py
uv run marimo edit code/notebooks/03b_task6_mortality.py
uv run marimo edit code/notebooks/03c_task7_readmission.py
```

### Headless Execution (All Steps)

``` bash
uv run marimo run code/notebooks/01_task_generator.py
uv run marimo run code/notebooks/02_feature_engineering.py
uv run python code/notebooks/02b_table_one.py
uv run marimo run code/notebooks/03a_task5_icu_los.py
uv run marimo run code/notebooks/03b_task6_mortality.py
uv run marimo run code/notebooks/03c_task7_readmission.py
```

### With Logging

``` bash
mkdir -p logs
uv run marimo run code/notebooks/01_task_generator.py 2>&1 | tee logs/01_tasks.log
uv run marimo run code/notebooks/02_feature_engineering.py 2>&1 | tee logs/02_features.log
uv run python code/notebooks/02b_table_one.py 2>&1 | tee logs/02b_table1.log
uv run marimo run code/notebooks/03b_task6_mortality.py 2>&1 | tee logs/03b_mortality.log
```

------------------------------------------------------------------------

## Troubleshooting

### Module Import Errors

Use interactive mode to debug:

``` bash
uv run marimo edit code/notebooks/01_task_generator.py
```

### Missing CLIF Tables

Check your `clif_config.json` data_directory path and verify required tables exist.

### Site Detection Issues

The notebooks detect your site from config:

``` python
SITE_NAME = config["site"]  # from clif_config.json
IS_RUSH = SITE_NAME.lower() == "rush"
IS_MIMIC = SITE_NAME.lower() == "mimic"
```

### Rush Models Not Found

For non-Rush sites, ensure `rush_models/` folder exists with models from BOX.