# FLAME-ICU

Federated Learning Adaptable Mortality Estimator for the ICU

## Project Overview

FLAME-ICU implements a multi-site federated learning approach for ICU mortality prediction using CLIF-standardized data. The project coordinates 6-7 institutions with RUSH as the main site, developing both XGBoost and Neural Network models.

### Stage 1: Model Development (3 Approaches)

1.  **Cross-Site Validation** - Test RUSH-trained models across sites without local training
2.  **Transfer Learning** - Fine-tune RUSH pre-trained models with local site data
3.  **Independent Training** - Each site trains models from scratch

### Stage 2: Comprehensive Testing

-   **Phase 1**: Cross-site testing of all Stage 1 models
-   **Phase 2**: Leave-one-out ensemble construction with accuracy weighting

### Data Split

-   **Training**: 2018-2022 admissions
-   **Validation**: 2023 admissions
-   **Testing**: 2024 admissions

### Deliverables

Models are shared via BOX for cross-site evaluation and ensemble construction, with final deployment recommendations based on performance metrics.

## Setup

### 1. Install UV

**Mac/Linux:**

``` bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**

``` powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Configure Site

Update `clif_config_template.json` and rename to `clif_config.json` and update fields:

``` json
{
    "site": "your_site_name",
    "data_directory": "/path/to/your/clif/data",
    "filetype": "parquet",
    "timezone": "US/Central"
}
```

### 3. Install Dependencies

``` bash
uv sync
```

**Troubleshooting:** If you encounter errors when running scripts with `uv run`, use `uv run marimo edit <script.py>` to debug at the cell level in an interactive environment.

## Required CLIF Tables

| Table | Columns | Categories |
|------------------|-----------------------|-------------------------------|
| **adt** | All columns | location_category |
| **hospitalization** | All columns | \- |
| **patient** | All columns | \- |
| **labs** | hospitalization_id, lab_result_dttm, lab_category, lab_value, lab_value_numeric | albumin, alt, ast, bicarbonate, bilirubin_total, bun, chloride, creatinine, inr, lactate, platelet_count, po2_arterial, potassium, pt, ptt, sodium, wbc |
| **vitals** | hospitalization_id, recorded_dttm, vital_category, vital_value | heart_rate, map, sbp, respiratory_rate, spo2, temp_c |
| **patient_assessments** | All columns | gcs_total |
| **medication_admin_continuous** | All columns (including med_dose, med_dose_unit) | norepinephrine, epinephrine, phenylephrine, vasopressin, dopamine, dobutamine, milrinone, isoproterenol |
| **respiratory_support** | All columns | device_category, fio2_set, peep_set |

------------------------------------------------------------------------

> **⚠️ IMPORTANT: Before Execution**
>
> **Visit CLIF BOX and download the `PHASE1_MODELS_UPLOAD_ME` folder, then place it in the project root directory.**
>
> This folder contains the necessary models required for the execution steps below.

------------------------------------------------------------------------

> **💡 OPTIONAL: Logging Command Output**
>
> To capture command output for debugging or record-keeping, you can append logging syntax to the end of any execution command:
>
> **Mac/Linux:** `2>&1 | tee logs/filename.log`
>
> **Windows:** `2>&1 | Tee-Object logs/filename.log`
>
> **Example:** `uv run code/preprocessing/01_cohort.py 2>&1 | tee logs/cohort.log`
>
> Create the logs directory first: `mkdir -p logs` (Mac/Linux) or `New-Item -ItemType Directory -Path logs` (Windows)

------------------------------------------------------------------------

## Execution Guide

### Prerequisites (All Approaches)

``` bash
# 1. Configure site (update clif_config.json)
# 2. Install dependencies
uv sync
# 3. Run preprocessing pipeline
# optional -> uv run code/preprocessing/00_scan_tables.py
uv run code/preprocessing/01_cohort.py
uv run code/preprocessing/02_feature_assmebly.py
uv run code/preprocessing/03_qc_heatmap.py
```

### Approach 1: Cross-Site Validation

**Federated Sites:**

``` bash
# Download RUSH models from BOX
# Visit CLIF BOX and download the model_storage folder, place it in project root
# Only run inference with RUSH models (no training)
uv run code/approach1_cross_site/stage_1/inference.py
# Upload results to BOX
```

### Approach 2: Transfer Learning

**Federated Sites:**

``` bash
# Download RUSH base models from BOX
# Visit CLIF BOX and download the model_storage folder, place it in project root
# After preprocessing, fine-tune RUSH models with local data
uv run code/approach2_transfer_learning/stage_1/transfer_learning.py
# Upload fine-tuned models to BOX
```

### Approach 3: Independent Training

**Federated Sites:**

``` bash
# After preprocessing, train models independently
uv run code/approach3_independent/stage_1/train_models.py
uv run code/approach3_independent/stage_1/inference.py
# Upload models to BOX
```

------------------------------------------------------------------------

## Uploading Results to BOX

**All Sites:**

After completing the execution steps for all approach, a `*_upload_me` folder will be generated in your project directory.

**Upload Instructions:**

-   Locate the `phase1_models_upload_me` & `phase1_results_upload_me` folder in your project root

-   Upload the entire folder to: `CLIF BOX / FLAME / [your_site_name]`

-   Replace `[your_site_name]` with your site name from `clif_config.json`

**Example:** upload to: `CLIF BOX / FLAME / your_site_name`

------------------------------------------------------------------------

> **⚠️ IMPORTANT: Stage 2 Coming - Preserve Your Data**
>
> **Stage 2 of this project is forthcoming and will require your preprocessed data files.**
>
> -   **DO NOT delete** your preprocessed data files after uploading results to BOX
> -   Keep all preprocessed data **safe and saved locally** in a secure location
> -   Ensure proper PHI data handling and security protocols are maintained
> -   You will need this data for Stage 2 execution