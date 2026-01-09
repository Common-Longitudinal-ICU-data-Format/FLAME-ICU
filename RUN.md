# FLAME-ICU Quick Start

## 1. Clone FLAME-ICU

``` bash
git clone https://github.com/Common-Longitudinal-ICU-data-Format/FLAME-ICU.git
cd FLAME-ICU
```

## 2. Clone FLAIR inside the FLAME repo

``` bash
git clone https://github.com/Common-Longitudinal-ICU-data-Format/FLAIR.git
```

## 3. Sync and install

``` bash
uv sync
uv pip install -e ./FLAIR
```

## 4. Download Rush models

Download the `rush_models/` folder from the CLIF BOX FLAME folder (link shared privately) and place it in the FLAME-ICU root directory.

## 5. Configure

Make sure your `clif_config.json` file is: - Properly renamed (from `clif_config_template.json`) - Updated with site paths and timezone

``` bash
cp clif_config_template.json clif_config.json
# Edit clif_config.json with your site info
```

------------------------------------------------------------------------

## Run the pipeline

### Step 1: Generate datasets

``` bash
uv run marimo edit code/notebooks/01_task_generator.py
```

### Step 2: Feature engineering

``` bash
uv run marimo edit code/notebooks/02_feature_engineering.py
```

### Step 2b: Generate Table 1

``` bash
uv run python code/notebooks/02b_table_one.py
```

### Step 3: Train & evaluate (run all three)

``` bash
uv run marimo edit code/notebooks/03a_task5_icu_los.py
uv run marimo edit code/notebooks/03b_task6_mortality.py
uv run marimo edit code/notebooks/03c_task7_readmission.py
```