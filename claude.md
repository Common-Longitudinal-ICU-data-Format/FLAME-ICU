# FLAIR Benchmark Implementation Plan

## Overview

Build prediction models for FLAIR benchmark tasks 5, 6, and 7 (ICU LOS, Hospital Mortality, ICU Readmission) using XGBoost and ElasticNet models.

## User Workflow

```bash
# Step 1: Install FLAIR library
cd FLAIR
uv pip install -e .

# Step 2: Back to main directory
cd ..

# Step 3: Create clif_config.json from template
cp FLAIR/clif_config_template.json clif_config.json
# Edit clif_config.json with your data paths

# Step 4: Run notebooks in order
marimo edit code/notebooks/01_task_generator.py      # Generate task datasets
marimo edit code/notebooks/02_feature_engineering.py # Extract features
marimo edit code/notebooks/03_train_evaluate.py      # Train and evaluate models
```

## Project Structure

```
FLAME-ICU/
├── old/                              # Previous code (renamed from code/)
├── code/
│   ├── __init__.py
│   ├── features.py                   # Feature extraction with clifpy
│   ├── notebooks/
│   │   ├── 01_task_generator.py      # Marimo: Generate datasets using flair library
│   │   ├── 02_feature_engineering.py # Marimo: Create features with clifpy
│   │   └── 03_train_evaluate.py      # Marimo: Train XGBoost + ElasticNet
│   └── models/
│       ├── __init__.py
│       ├── xgboost_model.py          # XGBoost wrapper
│       ├── elasticnet_model.py       # ElasticNet wrapper
│       └── evaluation.py             # Metrics computation
├── FLAIR/                            # FLAIR library (install with pip install -e .)
├── clif_config.json                  # Created from template (user data paths)
├── outputs/
│   ├── datasets/                     # Task datasets from 01
│   ├── features/                     # Feature datasets from 02
│   ├── models/                       # Trained models from 03
│   └── results/                      # Evaluation results from 03
└── claude.md                         # This file
```

## Task Details

| Task | Type | Input Window | Label | Metrics |
|------|------|--------------|-------|---------|
| 5 | Regression | 24hr | ICU LOS (hours) | MSE, RMSE, MAE, R2 |
| 6 | Binary Classification | 24hr | Mortality | AUROC, AUPRC, F1 |
| 7 | Binary Classification | Entire 1st ICU | Readmission | AUROC, AUPRC, F1 |

## FLAIR Library API

The FLAIR library provides `generate_task_dataset()` which handles:
- Cohort building from CLIF data
- Task-specific filtering
- Label extraction
- Time window calculation
- Temporal train/test splits

```python
from flair import generate_task_dataset

df = generate_task_dataset(
    config_path="clif_config.json",
    task_name="task5_icu_los",
    train_start="2020-01-01",
    train_end="2022-12-31",
    test_start="2023-01-01",
    test_end="2023-12-31",
    output_path="outputs/task5.parquet"
)
```

## Models

### XGBoost
- Optimized hyperparameters from previous work
- Native missing value handling
- Supports both regression (Task 5) and classification (Tasks 6, 7)

### ElasticNet
- L1+L2 regularization
- StandardScaler preprocessing
- Median imputation for missing values
- LogisticRegression for classification, ElasticNet for regression

## Feature Engineering

Features extracted from CLIF tables:
- **Vitals**: heart_rate, map, sbp, respiratory_rate, spo2, temp_c
- **Labs**: albumin, alt, ast, bicarbonate, bilirubin_total, bun, chloride, creatinine, inr, lactate, platelet_count, po2_arterial, potassium, pt, ptt, sodium, wbc
- **Respiratory Support**: device_category, fio2_set, peep_set
- **Medications**: vasopressor count
- **Patient Assessments**: gcs_total

Aggregation strategy:
- **Max**: lactate, bun, creatinine, inr, fio2_set, peep_set, heart_rate, temp_c, etc.
- **Min**: platelet_count, po2_arterial, spo2, sbp, map, albumin, etc.
- **Median**: respiratory_rate, fio2_set, map
- **Last**: gcs_total
- **Derived**: vasopressor_count, device one-hot encoding

## Configuration Files

### clif_config.json (create from template)
```json
{
  "data_sources": {
    "format": "parquet",
    "base_path": "./data/clif",
    "timezone": "US/Central"
  },
  "benchmark_outcomes": {
    "icu_los": {"generate_labels": true},
    "icu_readmission": {"generate_labels": true},
    "hospital_mortality": {"generate_labels": true}
  }
}
```

## Dependencies

Key dependencies (in pyproject.toml):
- flair-benchmark (local, from FLAIR/)
- clifpy >= 0.3.4
- xgboost
- scikit-learn
- polars
- pandas
- marimo
