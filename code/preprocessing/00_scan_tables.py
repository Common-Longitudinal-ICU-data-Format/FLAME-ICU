#!/usr/bin/env python3
"""
CLIF Table Validation Scanner - Focused Version
===============================================

Validates ONLY the specific columns used in the preprocessing pipeline.
Performs targeted validation for columns actually needed by 01_cohort.py and 02_feature_assmebly.py.

Key Features:
- Loads only required columns (memory efficient)
- Reports only critical errors that would break preprocessing
- Fast execution with minimal memory footprint
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import CLIF tables
from clifpy.tables import (
    Adt,
    Hospitalization,
    Patient,
    Labs,
    Vitals,
    PatientAssessments,
    MedicationAdminContinuous,
    RespiratorySupport
)


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def setup_logging(log_file: str = 'tables_check.log'):
    """
    Setup dual logging to both file and console.

    Args:
        log_file: Name of log file (will be created in project root)
    """
    # Get project root directory (2 levels up from this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    code_dir = os.path.dirname(script_dir)
    project_root = os.path.dirname(code_dir)
    log_path = os.path.join(project_root, log_file)

    # Remove existing handlers
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set root logger level
    logger.setLevel(logging.DEBUG)

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler - only INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return log_path


def print_colored(text: str, color: str = Colors.ENDC):
    """Print colored text to terminal."""
    print(f"{color}{text}{Colors.ENDC}")


def log_and_print(message: str, level: str = 'info', color: str = None):
    """
    Log message to file and print to console (with optional color).

    Args:
        message: Message to log/print
        level: Logging level ('debug', 'info', 'warning', 'error')
        color: ANSI color code for console output (optional)
    """
    # Log to file
    logger = logging.getLogger()
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)

    # Print to console with color (if not already printed by console handler)
    if color and level.lower() != 'info':
        print_colored(message, color)


def print_header(text: str):
    """Print a formatted header."""
    print()
    print_colored("=" * 80, Colors.HEADER)
    print_colored(text, Colors.HEADER + Colors.BOLD)
    print_colored("=" * 80, Colors.HEADER)

    # Also log header without color codes
    logging.info("=" * 80)
    logging.info(text)
    logging.info("=" * 80)


# Define ONLY the columns actually used in preprocessing
REQUIRED_COLUMNS = {
    'Patient': {
        'columns': ['patient_id', 'sex_category', 'race_category', 'ethnicity_category', 'language_category'],
        'critical': ['patient_id']  # Must have for pipeline to work
    },
    'Hospitalization': {
        'columns': ['patient_id', 'hospitalization_id', 'age_at_admission', 'discharge_category', 'admission_dttm'],
        'critical': ['hospitalization_id', 'admission_dttm', 'age_at_admission']
    },
    'Adt': {
        'columns': ['hospitalization_id', 'location_category', 'in_dttm', 'out_dttm'],
        'critical': ['hospitalization_id', 'location_category', 'in_dttm', 'out_dttm']
    },
    'Vitals': {
        'columns': ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        'critical': ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        'categories_used': ['heart_rate', 'map', 'sbp', 'respiratory_rate', 'spo2', 'temp_c', 'weight_kg']  # weight_kg used for med unit conversion
    },
    'Labs': {
        'columns': ['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value', 'lab_value_numeric'],
        'critical': ['hospitalization_id', 'lab_result_dttm', 'lab_category'],
        'categories_used': [
            'albumin', 'alt', 'ast', 'bicarbonate', 'bilirubin_total', 'bun', 'chloride',
            'creatinine', 'inr', 'lactate', 'platelet_count', 'po2_arterial', 'potassium',
            'pt', 'ptt', 'sodium', 'wbc'  # From 02_feature_assmebly.py category_filters
        ]
    },
    'PatientAssessments': {
        'columns': ['hospitalization_id', 'recorded_dttm', 'assessment_category', 'numerical_value'],
        'critical': ['hospitalization_id', 'assessment_category'],
        'categories_used': ['gcs_total']
    },
    'MedicationAdminContinuous': {
        'columns': None,  # Load all - needed for unit conversion
        'critical': ['hospitalization_id', 'admin_dttm', 'med_category'],
        'categories_used': [
            'norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin',
            'dopamine', 'dobutamine', 'milrinone', 'isoproterenol'
        ]
    },
    'RespiratorySupport': {
        'columns': None,  # Load all - multiple columns used
        'critical': ['hospitalization_id', 'recorded_dttm', 'device_category'],
        'categories_used': ['device_category', 'fio2_set', 'peep_set']
    }
}


def validate_required_columns(df: pd.DataFrame, table_name: str) -> Tuple[List[str], List[str]]:
    """
    Validate that required columns exist in the dataframe.

    Returns:
        Tuple of (missing_critical_columns, missing_optional_columns)
    """
    config = REQUIRED_COLUMNS.get(table_name, {})
    required_cols = config.get('columns', [])
    critical_cols = config.get('critical', [])

    if required_cols is None:  # Table loads all columns
        critical_cols = config.get('critical', [])
        missing_critical = [col for col in critical_cols if col not in df.columns]
        return missing_critical, []

    missing_critical = [col for col in critical_cols if col not in df.columns]
    missing_optional = [col for col in required_cols if col not in df.columns and col not in critical_cols]

    return missing_critical, missing_optional


def validate_categories(df: pd.DataFrame, table_name: str) -> Dict[str, List[str]]:
    """
    Check if required category values exist in the data.

    Returns:
        Dict with missing and available categories
    """
    config = REQUIRED_COLUMNS.get(table_name, {})
    categories_used = config.get('categories_used', [])

    if not categories_used:
        return {}

    # Determine the category column name
    if 'vital_category' in df.columns:
        cat_col = 'vital_category'
    elif 'lab_category' in df.columns:
        cat_col = 'lab_category'
    elif 'assessment_category' in df.columns:
        cat_col = 'assessment_category'
    elif 'med_category' in df.columns:
        cat_col = 'med_category'
    else:
        return {}

    # Get unique categories in the data
    available_categories = df[cat_col].dropna().unique().tolist()

    # Find missing categories
    missing_categories = [cat for cat in categories_used if cat not in available_categories]

    return {
        'missing': missing_categories,
        'available': available_categories,
        'required': categories_used
    }


def validate_table_focused(table_class, table_name: str, config_path: str = 'clif_config.json') -> Dict[str, Any]:
    """
    Load and validate a single CLIF table with focus on preprocessing requirements.

    Returns:
        Dict containing focused validation results
    """
    start_time = datetime.now()

    result = {
        'table_name': table_name,
        'status': 'not_checked',
        'critical_errors': [],
        'warnings': [],
        'stats': {}
    }

    try:
        # Get columns to load
        table_config = REQUIRED_COLUMNS.get(table_name, {})
        columns_to_load = table_config.get('columns')

        # Load table with only required columns
        print(f"  Loading {table_name}", end='')
        logging.info(f"Loading table: {table_name}")

        if columns_to_load:
            print(f" ({len(columns_to_load)} columns)", end='')
            logging.debug(f"  Columns to load: {columns_to_load}")
        else:
            print(f" (all columns)", end='')
            logging.debug(f"  Loading all columns for {table_name}")
        print("...", end='')

        load_start = datetime.now()
        table_instance = table_class.from_file(
            config_path=config_path,
            columns=columns_to_load,
            sample_size=None
        )
        load_time = (datetime.now() - load_start).total_seconds()

        if table_instance.df is None or len(table_instance.df) == 0:
            result['status'] = 'no_data'
            result['critical_errors'].append("No data loaded")
            print_colored(" [ERROR] No data", Colors.FAIL)
            logging.error(f"  {table_name}: No data loaded")
            return result

        df = table_instance.df
        result['stats']['num_records'] = len(df)
        result['stats']['num_columns'] = len(df.columns)
        result['stats']['load_time_seconds'] = round(load_time, 2)
        result['stats']['memory_mb'] = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)

        print_colored(f" [OK] {len(df):,} records", Colors.OKGREEN)
        logging.info(f"  Loaded {len(df):,} records in {load_time:.2f}s")
        logging.debug(f"  Memory usage: {result['stats']['memory_mb']:.2f} MB")
        logging.debug(f"  Columns present: {', '.join(df.columns.tolist())}")

        # Validate required columns
        logging.debug(f"  Validating required columns for {table_name}")
        missing_critical, missing_optional = validate_required_columns(df, table_name)

        if missing_critical:
            result['critical_errors'].append(f"Missing critical columns: {missing_critical}")
            print_colored(f"    [ERROR] Missing critical columns: {', '.join(missing_critical)}", Colors.FAIL)
            logging.error(f"  Missing critical columns: {', '.join(missing_critical)}")
        else:
            logging.debug(f"  All critical columns present")

        if missing_optional:
            result['warnings'].append(f"Missing optional columns: {missing_optional}")
            print_colored(f"    [WARNING] Missing optional columns: {', '.join(missing_optional)}", Colors.WARNING)
            logging.warning(f"  Missing optional columns: {', '.join(missing_optional)}")

        # Validate categories if applicable
        logging.debug(f"  Validating categories for {table_name}")
        category_check = validate_categories(df, table_name)
        if category_check:
            missing_cats = category_check.get('missing', [])
            available_cats = category_check.get('available', [])
            required_cats = category_check.get('required', [])

            logging.debug(f"  Required categories: {len(required_cats)}, Available: {len(available_cats)}, Missing: {len(missing_cats)}")
            if available_cats:
                logging.debug(f"  Available categories: {', '.join(available_cats[:10])}{' ...' if len(available_cats) > 10 else ''}")

            if missing_cats:
                # Only critical for feature tables
                if table_name in ['Vitals', 'Labs', 'PatientAssessments']:
                    result['critical_errors'].append(f"Missing required categories: {missing_cats[:5]}")
                    print_colored(f"    [ERROR] Missing categories: {', '.join(missing_cats[:5])}", Colors.FAIL)
                    logging.error(f"  Missing required categories: {', '.join(missing_cats)}")
                    if len(missing_cats) > 5:
                        print_colored(f"       ... and {len(missing_cats)-5} more", Colors.FAIL)
                else:
                    result['warnings'].append(f"Missing categories: {missing_cats[:5]}")
                    logging.warning(f"  Missing categories: {', '.join(missing_cats)}")

        # Check for null values in critical columns
        logging.debug(f"  Checking null values in critical columns")
        critical_cols = table_config.get('critical', [])
        for col in critical_cols:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                null_pct = (null_count / len(df)) * 100
                logging.debug(f"    {col}: {null_count:,} nulls ({null_pct:.1f}%)")

                if null_pct > 50:
                    result['critical_errors'].append(f"Column '{col}' has {null_pct:.1f}% missing values")
                    print_colored(f"    [ERROR] {col}: {null_pct:.1f}% missing", Colors.FAIL)
                    logging.error(f"  Column '{col}' has {null_pct:.1f}% missing values (critical)")
                elif null_pct > 10:
                    result['warnings'].append(f"Column '{col}' has {null_pct:.1f}% missing values")
                    logging.warning(f"  Column '{col}' has {null_pct:.1f}% missing values")

        # Log data types for critical columns
        logging.debug(f"  Data types for critical columns:")
        for col in critical_cols:
            if col in df.columns:
                logging.debug(f"    {col}: {df[col].dtype}")

        # Determine overall status
        if result['critical_errors']:
            result['status'] = 'error'
            logging.info(f"  Status: ERROR ({len(result['critical_errors'])} critical issues)")
        elif result['warnings']:
            result['status'] = 'warning'
            logging.info(f"  Status: WARNING ({len(result['warnings'])} warnings)")
        else:
            result['status'] = 'valid'
            logging.info(f"  Status: VALID")

        # Log total validation time
        total_time = (datetime.now() - start_time).total_seconds()
        result['stats']['total_validation_time_seconds'] = round(total_time, 2)
        logging.debug(f"  Total validation time: {total_time:.2f}s")

        # Clean up
        del table_instance

    except FileNotFoundError:
        result['status'] = 'not_found'
        result['critical_errors'].append("Table file not found")
        print_colored(" [ERROR] File not found", Colors.FAIL)
        logging.error(f"  {table_name}: Table file not found")

    except Exception as e:
        result['status'] = 'error'
        result['critical_errors'].append(f"Load error: {str(e)}")
        print_colored(f" [ERROR] {str(e)}", Colors.FAIL)
        logging.error(f"  {table_name}: Load error - {str(e)}")

    return result


def check_pipeline_readiness(results: List[Dict]) -> Dict[str, bool]:
    """
    Check if each preprocessing script can run based on validation results.
    """
    # Tables needed for 01_cohort.py
    cohort_tables = ['Patient', 'Hospitalization', 'Adt']
    sofa_tables = ['Labs', 'Vitals', 'PatientAssessments', 'MedicationAdminContinuous', 'RespiratorySupport']

    # Tables needed for 02_feature_assmebly.py
    feature_tables = ['Vitals', 'Labs', 'RespiratorySupport', 'MedicationAdminContinuous',
                     'PatientAssessments', 'Hospitalization']

    # Check readiness
    cohort_core_ready = all(
        any(r['table_name'] == t and r['status'] != 'error' and r['status'] != 'not_found' for r in results)
        for t in cohort_tables
    )

    sofa_ready = all(
        any(r['table_name'] == t and r['status'] != 'not_found' for r in results)
        for t in sofa_tables
    )

    features_ready = all(
        any(r['table_name'] == t and r['status'] != 'error' and r['status'] != 'not_found' for r in results)
        for t in feature_tables
    )

    return {
        'cohort_generation': cohort_core_ready,
        'sofa_calculation': sofa_ready,
        'feature_extraction': features_ready
    }


def main():
    """Main execution function."""
    # Setup logging first
    log_path = setup_logging('tables_check.log')

    # Record start time
    script_start_time = datetime.now()

    # Define tables to validate
    TABLES_TO_VALIDATE = [
        (Patient, 'Patient'),
        (Hospitalization, 'Hospitalization'),
        (Adt, 'Adt'),
        (Vitals, 'Vitals'),
        (Labs, 'Labs'),
        (PatientAssessments, 'PatientAssessments'),
        (MedicationAdminContinuous, 'MedicationAdminContinuous'),
        (RespiratorySupport, 'RespiratorySupport')
    ]

    print_header("CLIF TABLE VALIDATION - PREPROCESSING FOCUSED")
    print_colored("Validating only columns required for preprocessing pipeline", Colors.OKBLUE)
    logging.info("Validating only columns required for preprocessing pipeline")
    logging.info(f"Log file: {log_path}")

    # Load config
    try:
        with open('clif_config.json', 'r') as f:
            config = json.load(f)
        print(f"\nSite: {config.get('site', 'unknown')}")
        print(f"Data: {config.get('data_directory', 'unknown')}")
        print(f"Type: {config.get('filetype', 'unknown')}")

        logging.info(f"Site: {config.get('site', 'unknown')}")
        logging.info(f"Data directory: {config.get('data_directory', 'unknown')}")
        logging.info(f"File type: {config.get('filetype', 'unknown')}")
        logging.info(f"Timezone: {config.get('timezone', 'unknown')}")
        logging.debug(f"Full config: {json.dumps(config, indent=2)}")
    except Exception as e:
        print_colored(f"Warning: Could not load config: {e}", Colors.WARNING)
        logging.warning(f"Could not load config: {e}")

    print_colored("\n" + "─" * 80, Colors.OKCYAN)
    logging.info("─" * 80)

    # Validate each table
    results = []
    error_count = 0
    warning_count = 0
    total_records = 0

    logging.info(f"Starting validation of {len(TABLES_TO_VALIDATE)} tables")

    for i, (table_class, table_name) in enumerate(TABLES_TO_VALIDATE, 1):
        print(f"\n[{i}/{len(TABLES_TO_VALIDATE)}] {table_name}")
        logging.info(f"\n[{i}/{len(TABLES_TO_VALIDATE)}] {table_name}")
        logging.info("─" * 40)

        result = validate_table_focused(table_class, table_name)
        results.append(result)

        if result['critical_errors']:
            error_count += len(result['critical_errors'])
        if result['warnings']:
            warning_count += len(result['warnings'])
        if result.get('stats', {}).get('num_records'):
            total_records += result['stats']['num_records']

    # Calculate total execution time
    total_execution_time = (datetime.now() - script_start_time).total_seconds()

    # Summary
    print_header("VALIDATION SUMMARY")
    logging.info("")
    logging.info("=" * 80)

    # Count statuses
    valid_tables = sum(1 for r in results if r['status'] == 'valid')
    warning_tables = sum(1 for r in results if r['status'] == 'warning')
    error_tables = sum(1 for r in results if r['status'] == 'error')
    not_found_tables = sum(1 for r in results if r['status'] == 'not_found')

    print(f"[OK] Valid: {valid_tables}/{len(results)} tables")
    print(f"[WARNING] Warnings: {warning_tables}/{len(results)} tables ({warning_count} issues)")
    print(f"[ERROR] Errors: {error_tables}/{len(results)} tables ({error_count} critical issues)")

    logging.info(f"VALIDATION SUMMARY")
    logging.info(f"Valid tables: {valid_tables}/{len(results)}")
    logging.info(f"Tables with warnings: {warning_tables}/{len(results)} ({warning_count} total warnings)")
    logging.info(f"Tables with errors: {error_tables}/{len(results)} ({error_count} critical errors)")

    if not_found_tables > 0:
        print(f"[INFO] Not Found: {not_found_tables} tables")
        logging.info(f"Tables not found: {not_found_tables}")

    # Log detailed statistics
    logging.info("")
    logging.info("DETAILED STATISTICS:")
    logging.info(f"Total execution time: {total_execution_time:.2f}s")
    logging.info(f"Total records processed: {total_records:,}")

    # Log per-table statistics
    logging.debug("")
    logging.debug("PER-TABLE STATISTICS:")
    for r in results:
        if r.get('stats'):
            stats = r['stats']
            logging.debug(f"  {r['table_name']}:")
            logging.debug(f"    Records: {stats.get('num_records', 'N/A'):,}")
            logging.debug(f"    Columns: {stats.get('num_columns', 'N/A')}")
            logging.debug(f"    Load time: {stats.get('load_time_seconds', 'N/A')}s")
            logging.debug(f"    Memory: {stats.get('memory_mb', 'N/A')} MB")
            logging.debug(f"    Status: {r['status']}")

    # Pipeline readiness
    print_colored("\n" + "─" * 80, Colors.OKCYAN)
    print_colored("PREPROCESSING PIPELINE READINESS", Colors.BOLD)
    print()

    logging.info("")
    logging.info("─" * 80)
    logging.info("PREPROCESSING PIPELINE READINESS")
    logging.info("")

    readiness = check_pipeline_readiness(results)

    # 01_cohort.py readiness
    if readiness['cohort_generation']:
        print_colored("[OK] 01_cohort.py - Cohort Generation: READY", Colors.OKGREEN)
        logging.info("01_cohort.py - Cohort Generation: READY")
    else:
        print_colored("[ERROR] 01_cohort.py - Cohort Generation: NOT READY", Colors.FAIL)
        logging.error("01_cohort.py - Cohort Generation: NOT READY")
        # Show which tables are blocking
        cohort_tables = ['Patient', 'Hospitalization', 'Adt']
        for t in cohort_tables:
            for r in results:
                if r['table_name'] == t and r['status'] in ['error', 'not_found']:
                    msg = f"  Blocking table: {t} - {r['critical_errors'][0] if r['critical_errors'] else 'Not found'}"
                    print(f"     └─ {t}: {r['critical_errors'][0] if r['critical_errors'] else 'Not found'}")
                    logging.error(msg)

    if readiness['sofa_calculation']:
        print_colored("[OK] 01_cohort.py - SOFA Calculation: READY", Colors.OKGREEN)
        logging.info("01_cohort.py - SOFA Calculation: READY")
    else:
        print_colored("[WARNING] 01_cohort.py - SOFA Calculation: PARTIAL", Colors.WARNING)
        logging.warning("01_cohort.py - SOFA Calculation: PARTIAL")
        print("     └─ SOFA scores may be incomplete due to missing tables")
        logging.warning("  SOFA scores may be incomplete due to missing tables")

    # 02_feature_assmebly.py readiness
    if readiness['feature_extraction']:
        print_colored("[OK] 02_feature_assmebly.py - Feature Extraction: READY", Colors.OKGREEN)
        logging.info("02_feature_assmebly.py - Feature Extraction: READY")
    else:
        print_colored("[ERROR] 02_feature_assmebly.py - Feature Extraction: NOT READY", Colors.FAIL)
        logging.error("02_feature_assmebly.py - Feature Extraction: NOT READY")
        # Show which tables are blocking
        feature_tables = ['Vitals', 'Labs', 'RespiratorySupport', 'MedicationAdminContinuous', 'PatientAssessments', 'Hospitalization']
        for t in feature_tables:
            for r in results:
                if r['table_name'] == t and r['status'] in ['error', 'not_found']:
                    msg = f"  Blocking table: {t} - {r['critical_errors'][0] if r['critical_errors'] else 'Not found'}"
                    print(f"     └─ {t}: {r['critical_errors'][0] if r['critical_errors'] else 'Not found'}")
                    logging.error(msg)

    # Critical issues summary
    if error_count > 0:
        print_colored("\n" + "─" * 80, Colors.OKCYAN)
        print_colored("[WARNING] CRITICAL ISSUES TO RESOLVE", Colors.FAIL + Colors.BOLD)
        print()

        logging.info("")
        logging.info("─" * 80)
        logging.info("CRITICAL ISSUES TO RESOLVE")
        logging.info("")

        for r in results:
            if r['critical_errors']:
                print_colored(f"• {r['table_name']}:", Colors.FAIL)
                logging.error(f"{r['table_name']} critical errors:")
                for error in r['critical_errors'][:3]:  # Show max 3 errors per table
                    print(f"  - {error}")
                    logging.error(f"  - {error}")
                # Log all errors to file (not just first 3)
                if len(r['critical_errors']) > 3:
                    for error in r['critical_errors'][3:]:
                        logging.error(f"  - {error}")

    print_colored("\n" + "=" * 80, Colors.HEADER)

    # Final summary to log
    logging.info("")
    logging.info("=" * 80)
    logging.info("VALIDATION COMPLETE")
    logging.info(f"Execution time: {total_execution_time:.2f}s")
    logging.info(f"Exit code: {1 if (error_tables > 0 or not_found_tables > 0) else 0}")
    logging.info("=" * 80)

    # Inform user about log location
    print(f"\nDetailed log saved to: {log_path}")

    # Return exit code
    if error_tables > 0 or not_found_tables > 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    sys.exit(main())