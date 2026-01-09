"""
Feature extraction and aggregation for FLAIR benchmark tasks.

Uses clifpy to load CLIF tables and create aggregated features.
Reuses patterns from old/preprocessing/02_feature_assmebly.py
"""

import json
import pandas as pd
import numpy as np
import polars as pl
from typing import Dict, List, Any, Optional
from clifpy.clif_orchestrator import ClifOrchestrator
from clifpy.utils.outlier_handler import apply_outlier_handling


# Feature configuration
CATEGORY_FILTERS = {
    'vitals': [
        'heart_rate', 'map', 'sbp', 'respiratory_rate', 'spo2', 'temp_c'
    ],
    'labs': [
        "albumin", "alt", "ast", "bicarbonate", "bilirubin_total", "bun", "chloride", "creatinine",
        "inr", "lactate", "platelet_count", "po2_arterial", "potassium", "pt", "ptt",
        "sodium", "wbc"
    ],
    'medication_admin_continuous': [
        "norepinephrine", "epinephrine", "phenylephrine", "vasopressin",
        "dopamine", "dobutamine", "milrinone", "isoproterenol"
    ],
    'respiratory_support': [
        'device_category', 'fio2_set', 'peep_set'
    ],
    'patient_assessments': [
        'gcs_total'
    ]
}

# Aggregation configuration
AGGREGATION_CONFIG = {
    'max_features': [
        'lactate', 'bun', 'creatinine', 'ast', 'alt', 'bilirubin_total',
        'inr', 'ptt', 'pt', 'fio2_set', 'peep_set', 'respiratory_rate',
        'heart_rate', 'temp_c', 'sodium', 'potassium', 'wbc', 'chloride', 'bicarbonate'
    ],
    'min_features': [
        'platelet_count', 'po2_arterial', 'spo2', 'sbp', 'map', 'albumin',
        'sodium', 'potassium', 'wbc', 'temp_c', 'heart_rate'
    ],
    'median_features': [
        'respiratory_rate', 'fio2_set', 'bilirubin_total', 'map'
    ],
    'last_features': [
        'gcs_total'
    ]
}

# Devices to one-hot encode (exclude Room Air and Other)
DEVICES_TO_INCLUDE = ['imv', 'nippv', 'cpap', 'high flow nc', 'face mask', 'trach collar', 'nasal cannula']

# Vasopressor categories for count
VASOPRESSOR_CATEGORIES = [
    'norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin',
    'dopamine', 'dobutamine', 'milrinone', 'isoproterenol'
]


class FeatureExtractor:
    """
    Feature extraction and aggregation for FLAIR benchmark tasks.

    Uses clifpy to load CLIF tables and create wide datasets.
    Applies outlier handling and aggregation patterns.
    """

    def __init__(self, config_path: str):
        """
        Initialize with CLIF config path.

        Args:
            config_path: Path to clif_config.json
        """
        self.config_path = config_path
        self.clif = ClifOrchestrator(config_path=config_path)
        self.wide_df = None

    def load_tables(self, cohort_ids: List[str]) -> None:
        """
        Load required CLIF tables with cohort and category filtering.

        Args:
            cohort_ids: List of hospitalization_id values to filter
        """
        print(f"Loading CLIF tables for {len(cohort_ids)} hospitalizations...")

        table_config = {
            'vitals': {
                'columns': ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
                'category_filter': ('vital_category', CATEGORY_FILTERS['vitals'])
            },
            'labs': {
                'columns': ['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value', 'lab_value_numeric'],
                'category_filter': ('lab_category', CATEGORY_FILTERS['labs'])
            },
            'respiratory_support': {
                'columns': None,
                'category_filter': None
            },
            'medication_admin_continuous': {
                'columns': None,
                'category_filter': ('med_category', CATEGORY_FILTERS['medication_admin_continuous'])
            },
            'patient_assessments': {
                'columns': None,
                'category_filter': ('assessment_category', CATEGORY_FILTERS['patient_assessments'])
            }
        }

        for table_name, tbl_config in table_config.items():
            print(f"  Loading {table_name}...")
            filters_dict = {'hospitalization_id': cohort_ids}

            if tbl_config.get('category_filter'):
                cat_col, cat_values = tbl_config['category_filter']
                if cat_values:
                    filters_dict[cat_col] = cat_values

            try:
                self.clif.load_table(
                    table_name,
                    filters=filters_dict,
                    columns=tbl_config.get('columns')
                )
                print(f"    Loaded {table_name}")
            except Exception as e:
                print(f"    Warning: Could not load {table_name}: {e}")

    def apply_outlier_handling(self) -> None:
        """Apply clifpy outlier handling to loaded tables."""
        print("Applying outlier handling...")
        for table_name in ['vitals', 'labs', 'respiratory_support', 'patient_assessments']:
            try:
                table_obj = getattr(self.clif, table_name, None)
                if table_obj is not None and hasattr(table_obj, 'df') and table_obj.df is not None:
                    apply_outlier_handling(table_obj)
                    print(f"  Applied to {table_name}")
            except Exception as e:
                print(f"  Warning: Could not apply to {table_name}: {e}")

    def create_wide_dataset(
        self,
        cohort_df: pd.DataFrame,
        time_col_start: str = 'window_start',
        time_col_end: str = 'window_end'
    ) -> pd.DataFrame:
        """
        Create wide dataset filtered by time windows.

        Args:
            cohort_df: DataFrame with hospitalization_id, window_start, window_end columns
            time_col_start: Column name for window start time
            time_col_end: Column name for window end time

        Returns:
            Wide format DataFrame with features
        """
        print("Creating wide dataset...")

        # Prepare time filter DataFrame for clifpy
        cohort_time_filter = cohort_df[['hospitalization_id', time_col_start, time_col_end]].copy()
        cohort_time_filter.columns = ['hospitalization_id', 'start_time', 'end_time']

        # Convert to pandas if polars
        if isinstance(cohort_time_filter, pl.DataFrame):
            cohort_time_filter = cohort_time_filter.to_pandas()

        self.clif.create_wide_dataset(
            category_filters=CATEGORY_FILTERS,
            cohort_df=cohort_time_filter,
            save_to_data_location=False,
            batch_size=10000,
            memory_limit='6GB',
            threads=4,
            show_progress=True
        )

        self.wide_df = self.clif.wide_df.copy()
        print(f"Wide dataset shape: {self.wide_df.shape}")
        return self.wide_df

    def aggregate_features(self, wide_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Aggregate event-level data to one row per hospitalization.

        Args:
            wide_df: Wide format DataFrame (uses self.wide_df if not provided)

        Returns:
            Aggregated DataFrame with one row per hospitalization
        """
        if wide_df is None:
            wide_df = self.wide_df

        if wide_df is None:
            raise ValueError("No wide dataset available. Call create_wide_dataset first.")

        print("Aggregating features...")

        # Build aggregation dictionary
        agg_dict = {}

        # Max aggregations
        for feat in AGGREGATION_CONFIG['max_features']:
            if feat in wide_df.columns:
                agg_dict[f'{feat}_max'] = pd.NamedAgg(column=feat, aggfunc='max')

        # Min aggregations
        for feat in AGGREGATION_CONFIG['min_features']:
            if feat in wide_df.columns:
                agg_dict[f'{feat}_min'] = pd.NamedAgg(column=feat, aggfunc='min')

        # Median aggregations
        for feat in AGGREGATION_CONFIG['median_features']:
            if feat in wide_df.columns:
                agg_dict[f'{feat}_median'] = pd.NamedAgg(column=feat, aggfunc='median')

        # Last aggregations (most recent value)
        for feat in AGGREGATION_CONFIG['last_features']:
            if feat in wide_df.columns:
                agg_dict[f'{feat}_last'] = pd.NamedAgg(column=feat, aggfunc='last')

        # Sort by time and aggregate
        if 'event_time' in wide_df.columns:
            wide_df_sorted = wide_df.sort_values(['hospitalization_id', 'event_time'])
        else:
            wide_df_sorted = wide_df.sort_values(['hospitalization_id'])

        aggregated_df = wide_df_sorted.groupby('hospitalization_id', as_index=False).agg(**agg_dict)

        # Add device one-hot encoding
        aggregated_df = self._add_device_encoding(aggregated_df, wide_df)

        # Add vasopressor count
        aggregated_df = self._add_vasopressor_count(aggregated_df, wide_df)

        print(f"Aggregated shape: {aggregated_df.shape}")
        return aggregated_df

    def _add_device_encoding(self, aggregated_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
        """Add one-hot encoding for respiratory devices."""
        if 'device_category' not in wide_df.columns:
            return aggregated_df

        print("  Adding device one-hot encoding...")

        # Get device usage per hospitalization
        device_usage = wide_df.groupby('hospitalization_id')['device_category'].apply(
            lambda x: x.dropna().unique().tolist()
        ).reset_index()

        # One-hot encode each device
        for device in DEVICES_TO_INCLUDE:
            device_col = f'device_{device.replace(" ", "_")}'
            device_usage[device_col] = device_usage['device_category'].apply(
                lambda devices: 1 if any(str(d).lower() == device for d in devices) else 0
            )

        device_usage = device_usage.drop(columns=['device_category'])
        return pd.merge(aggregated_df, device_usage, on='hospitalization_id', how='left')

    def _add_vasopressor_count(self, aggregated_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
        """Add vasopressor count derived feature."""
        print("  Adding vasopressor count...")

        # Find vasopressor columns in wide_df
        vaso_cols = [col for col in wide_df.columns if col in VASOPRESSOR_CATEGORIES]

        if not vaso_cols:
            aggregated_df['vasopressor_count'] = 0
            return aggregated_df

        # Count unique vasopressors used per hospitalization
        vaso_count = wide_df.groupby('hospitalization_id').apply(
            lambda x: sum(x[vaso_cols].notna().any())
        ).reset_index(name='vasopressor_count')

        return pd.merge(aggregated_df, vaso_count, on='hospitalization_id', how='left')

    def add_demographics(
        self,
        features_df: pd.DataFrame,
        cohort_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add demographic features from cohort.

        Args:
            features_df: Aggregated features DataFrame
            cohort_df: Cohort DataFrame with demographics

        Returns:
            Features DataFrame with demographics added
        """
        print("Adding demographics...")

        # Convert polars to pandas if needed
        if isinstance(cohort_df, pl.DataFrame):
            cohort_df = cohort_df.to_pandas()

        demo_cols = ['hospitalization_id', 'age_at_admission', 'sex_category', 'race_category', 'ethnicity_category']
        available_cols = [c for c in demo_cols if c in cohort_df.columns]

        if len(available_cols) > 1:
            demographics = cohort_df[available_cols].drop_duplicates()
            features_df = pd.merge(features_df, demographics, on='hospitalization_id', how='left')

            # Add derived features
            if 'age_at_admission' in features_df.columns:
                features_df['age'] = features_df['age_at_admission']
            if 'sex_category' in features_df.columns:
                features_df['isfemale'] = (features_df['sex_category'].str.lower() == 'female').astype(int)

        return features_df


def extract_features_for_task(
    config_path: str,
    task_dataset: pd.DataFrame,
    time_col_start: str = 'window_start',
    time_col_end: str = 'window_end'
) -> pd.DataFrame:
    """
    Convenience function to extract features for a task dataset.

    Args:
        config_path: Path to clif_config.json
        task_dataset: Task dataset with hospitalization_id, window_start, window_end columns
        time_col_start: Column name for window start
        time_col_end: Column name for window end

    Returns:
        Aggregated features DataFrame
    """
    # Convert polars to pandas if needed
    if isinstance(task_dataset, pl.DataFrame):
        task_dataset = task_dataset.to_pandas()

    # Load timezone from config and localize datetime columns
    # FLAIR strips timezone info, so we need to add it back for clifpy compatibility
    with open(config_path) as f:
        config = json.load(f)
    timezone = config["timezone"]

    # Localize datetime columns (not convert - just add timezone info)
    # Using same logic as clifpy/utils/io.py for DST handling
    for col in [time_col_start, time_col_end]:
        if col in task_dataset.columns:
            if pd.api.types.is_datetime64_any_dtype(task_dataset[col]):
                if task_dataset[col].dt.tz is None:
                    task_dataset[col] = task_dataset[col].dt.tz_localize(
                        timezone, ambiguous=True, nonexistent='shift_forward'
                    )

    extractor = FeatureExtractor(config_path)

    # Get unique hospitalization IDs
    cohort_ids = task_dataset['hospitalization_id'].astype(str).unique().tolist()

    # Load tables
    extractor.load_tables(cohort_ids)

    # Apply outlier handling
    extractor.apply_outlier_handling()

    # Create wide dataset
    extractor.create_wide_dataset(task_dataset, time_col_start, time_col_end)

    # Aggregate features
    features_df = extractor.aggregate_features()

    # Add demographics
    features_df = extractor.add_demographics(features_df, task_dataset)

    return features_df
