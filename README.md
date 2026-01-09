# FLAME-ICU

**F**ederated **L**earning **A**daptable **M**ortality **E**stimator for the ICU

A multi-site federated learning implementation of the [FLAIR benchmark](./FLAIR/) for ICU outcome prediction using CLIF-standardized data.

------------------------------------------------------------------------

## What is FLAIR?

> **Note:** FLAIR is currently in development and will soon be available via PyPI as `flair-benchmark`.

FLAIR (Federated Learning for AI Research) is a privacy-preserving benchmark framework for ICU prediction tasks. It provides:

-   **7 clinically relevant prediction tasks** for ICU outcomes
-   **Standardized data format** using CLIF (Common Longitudinal ICU Format)
-   **Privacy-preserving design** - no data leaves your institution
-   **Reproducible benchmarks** across 17+ US hospitals

### FLAIR Tasks

| Task  | Name                   | Type               | Prediction Window   |
|-------|------------------------|--------------------|---------------------|
| **5** | **ICU Length of Stay** | **Regression**     | **24 hours**        |
| **6** | **Hospital Mortality** | **Classification** | **24 hours**        |
| **7** | **ICU Readmission**    | **Classification** | **Entire ICU stay** |

FLAME-ICU implements **Tasks 5, 6, and 7**.

------------------------------------------------------------------------

## Quick Start

### 1. Clone FLAME-ICU

``` bash
git clone https://github.com/Common-Longitudinal-ICU-data-Format/FLAME-ICU.git
cd FLAME-ICU
```

### 2. Clone FLAIR inside the FLAME repo

``` bash
git clone https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR.git
```

### 3. Sync and install

``` bash
uv sync
uv pip install -e ./FLAIR
```

### 4. Configure

Make sure your `clif_config.json` file is properly renamed and updated with site paths and timezone.

### Run the pipeline

``` bash
# Step 1: Generate datasets
uv run marimo edit code/notebooks/01_task_generator.py

# Step 2: Feature engineering
uv run marimo edit code/notebooks/02_feature_engineering.py

# Step 2b: Generate Table 1
uv run python code/notebooks/02b_table_one.py

# Step 3: Train & evaluate (run all three)
uv run marimo edit code/notebooks/03a_task5_icu_los.py
uv run marimo edit code/notebooks/03b_task6_mortality.py
uv run marimo edit code/notebooks/03c_task7_readmission.py
```

See [**RUN.md**](./RUN.md) for details.

------------------------------------------------------------------------

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

------------------------------------------------------------------------

## Federated Learning Workflow

### Rush Site (Baseline)

1.  Train XGBoost and ElasticNet models on Rush data
2.  Upload trained models to BOX
3.  Models serve as baseline for other sites

### Other Sites (Evaluation)

1.  Download Rush models from BOX
2.  Run 4 experiments per task:
    -   **Rush Evaluation** - Test Rush models on local data
    -   **Platt Scaling** - Calibrate Rush models with local train data
    -   **Transfer Learning** - Fine-tune Rush models with local data
    -   **Independent** - Train from scratch on local data
3.  Upload results to BOX

------------------------------------------------------------------------

## Models

### XGBoost

-   Gradient boosting with native missing value handling
-   Optimized hyperparameters via Optuna (50 trials, 3-fold CV)

### ElasticNet

-   L1+L2 regularized linear model
-   StandardScaler + median imputation
-   LogisticRegression for classification, ElasticNet for regression

------------------------------------------------------------------------

## Required CLIF Tables

| Table | Key Columns | Categories |
|------------------|----------------------------|--------------------------|
| hospitalization | All | \- |
| patient | All | \- |
| adt | All | location_category |
| labs | lab_result_dttm, lab_category, lab_value | albumin, creatinine, lactate, etc. |
| vitals | recorded_dttm, vital_category, vital_value | heart_rate, map, sbp, spo2, temp_c |
| patient_assessments | All | gcs_total |
| medication_admin_continuous | med_dose, med_dose_unit | vasopressors |
| respiratory_support | All | device_category, fio2_set, peep_set |

------------------------------------------------------------------------

## Configuration

Create `clif_config.json` from template:

``` json
{
    "site": "your_site_name",
    "data_directory": "/path/to/clif/data",
    "filetype": "parquet",
    "timezone": "US/Central"
}
```

------------------------------------------------------------------------

## Metrics

### Classification (Tasks 6, 7)

-   AUROC, AUPRC, F1, Accuracy
-   Precision, Recall, Specificity, NPV
-   Brier Score, ICI (Integrated Calibration Index)
-   DCA (Decision Curve Analysis)
-   Bootstrap 95% confidence intervals

### Regression (Task 5)

-   MSE, RMSE, MAE, R²
-   Predicted vs Observed plots

------------------------------------------------------------------------

## Dependencies

-   Python 3.11+
-   [UV](https://github.com/astral-sh/uv) package manager
-   [clifpy](https://github.com/clif-consortium/clifpy) \>= 0.3.4
-   [marimo](https://marimo.io/) notebooks
-   XGBoost, scikit-learn, pandas, polars

------------------------------------------------------------------------

## Support

For issues or questions: - Check [RUN.md](./RUN.md) troubleshooting section - Contact the FLAME-ICU team via CLIF consortium channels