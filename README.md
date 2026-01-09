# FLAME-ICU

**F**ederated **L**earning **A**daptable **M**ortality **E**stimator for the ICU

A multi-site federated learning implementation of the [FLAIR benchmark](./FLAIR/) for ICU outcome prediction using CLIF-standardized data.

---

## What is FLAIR?

> **Note:** FLAIR is currently in development and will soon be available via PyPI as `flair-benchmark`.

FLAIR (Federated Learning for AI Research) is a privacy-preserving benchmark framework for ICU prediction tasks. It provides:

- **7 clinically relevant prediction tasks** for ICU outcomes
- **Standardized data format** using CLIF (Common Longitudinal ICU Format)
- **Privacy-preserving design** - no data leaves your institution
- **Reproducible benchmarks** across 17+ US hospitals

### FLAIR Tasks

| Task | Name | Type | Prediction Window |
|------|------|------|-------------------|
| 1 | Discharged Home | Classification | 24 hours |
| 2 | Discharged to LTACH | Classification | 24 hours |
| 3 | 72-Hour Respiratory Outcome | Multiclass | Variable |
| 4 | Hypoxic Proportion | Regression | 24-72 hours |
| **5** | **ICU Length of Stay** | **Regression** | **24 hours** |
| **6** | **Hospital Mortality** | **Classification** | **24 hours** |
| **7** | **ICU Readmission** | **Classification** | **Entire ICU stay** |

FLAME-ICU implements **Tasks 5, 6, and 7**.

---

## Quick Start

See **[RUN.md](./RUN.md)** for complete step-by-step instructions.

### TL;DR

```bash
# 1. Install FLAIR (current - from local source)
cd FLAIR && uv pip install -e . && cd ..

# 1. Install FLAIR (future - when available on PyPI)
# uv pip install flair-benchmark

# 2. Install project dependencies
uv sync

# 3. Configure
cp clif_config_template.json clif_config.json
# Edit with your site name and data path

# 4. Run notebooks in order
uv run marimo edit code/notebooks/01_task_generator.py
uv run marimo edit code/notebooks/02_feature_engineering.py
uv run marimo edit code/notebooks/03b_task6_mortality.py
uv run marimo edit code/notebooks/03c_task7_readmission.py
```

---

## Project Structure

```
FLAME-ICU/
├── FLAIR/                    # FLAIR benchmark library
│   └── flair/                # Core library code
├── code/
│   ├── notebooks/            # Marimo notebooks (main workflow)
│   │   ├── 01_task_generator.py      # Generate task datasets
│   │   ├── 02_feature_engineering.py # Extract CLIF features
│   │   ├── 03a_task5_icu_los.py      # ICU LOS models
│   │   ├── 03b_task6_mortality.py    # Mortality models
│   │   └── 03c_task7_readmission.py  # Readmission models
│   ├── models/               # Model implementations
│   │   ├── xgboost_model.py
│   │   ├── elasticnet_model.py
│   │   └── evaluation.py
│   └── features.py           # Feature extraction
├── outputs/                  # Generated datasets and features
├── rush_models/              # Rush baseline models
├── results_to_box/           # Site results for upload
├── optimization/             # Hyperparameter tuning (Optuna)
├── clif_config.json          # Site configuration
├── RUN.md                    # Execution guide
└── README.md                 # This file
```

---

## Federated Learning Workflow

### Rush Site (Baseline)

1. Train XGBoost and ElasticNet models on Rush data
2. Upload trained models to BOX
3. Models serve as baseline for other sites

### Other Sites (Evaluation)

1. Download Rush models from BOX
2. Run 4 experiments per task:
   - **Rush Evaluation** - Test Rush models on local data
   - **Platt Scaling** - Calibrate Rush models with local train data
   - **Transfer Learning** - Fine-tune Rush models with local data
   - **Independent** - Train from scratch on local data
3. Upload results to BOX

---

## Models

### XGBoost
- Gradient boosting with native missing value handling
- Optimized hyperparameters via Optuna (50 trials, 3-fold CV)

### ElasticNet
- L1+L2 regularized linear model
- StandardScaler + median imputation
- LogisticRegression for classification, ElasticNet for regression

### Performance (Optuna-optimized)

| Model | Task 5 (R²) | Task 7 (AUROC) |
|-------|-------------|----------------|
| XGBoost | 0.183 | 0.654 |
| ElasticNet | 0.128 | 0.609 |

---

## Required CLIF Tables

| Table | Key Columns | Categories |
|-------|-------------|------------|
| hospitalization | All | - |
| patient | All | - |
| adt | All | location_category |
| labs | lab_result_dttm, lab_category, lab_value | albumin, creatinine, lactate, etc. |
| vitals | recorded_dttm, vital_category, vital_value | heart_rate, map, sbp, spo2, temp_c |
| patient_assessments | All | gcs_total |
| medication_admin_continuous | med_dose, med_dose_unit | vasopressors |
| respiratory_support | All | device_category, fio2_set, peep_set |

---

## Configuration

Create `clif_config.json` from template:

```json
{
    "site": "your_site_name",
    "data_directory": "/path/to/clif/data",
    "filetype": "parquet",
    "timezone": "US/Central"
}
```

---

## Metrics

### Classification (Tasks 6, 7)
- AUROC, AUPRC, F1, Accuracy
- Precision, Recall, Specificity, NPV
- Brier Score, ICI (Integrated Calibration Index)
- DCA (Decision Curve Analysis)
- Bootstrap 95% confidence intervals

### Regression (Task 5)
- MSE, RMSE, MAE, R²
- Predicted vs Observed plots

---

## Dependencies

- Python 3.11+
- [UV](https://github.com/astral-sh/uv) package manager
- [clifpy](https://github.com/clif-consortium/clifpy) >= 0.3.4
- [marimo](https://marimo.io/) notebooks
- XGBoost, scikit-learn, pandas, polars

---

## Privacy & Compliance

FLAIR is designed for privacy-preserving federated learning:

- No network requests allowed during execution
- PHI detection scans all outputs
- Cell counts < 10 are suppressed (HIPAA safe harbor)
- Code reviewed by institutional PIs before execution
- Individual-level data never leaves the institution

---

## Support

For issues or questions:
- Check [RUN.md](./RUN.md) troubleshooting section
- Contact the FLAME-ICU team via CLIF consortium channels
