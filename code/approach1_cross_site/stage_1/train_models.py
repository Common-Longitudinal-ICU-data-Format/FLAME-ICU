#!/usr/bin/env python3
"""
Approach 1 Stage 1: Train Models for Cross-Site Validation

This script implements Approach 1 (Cross-Site Model Validation) for Stage 1:
- Trains XGBoost and Neural Network models on RUSH training data
- Uses validation set for early stopping and hyperparameter validation
- Saves trained models for distribution to other sites
- No local training required at federated sites

Models trained here will be shared with other sites for testing in Stage 2.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

# Add parent directories to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
approach_dir = os.path.dirname(script_dir)  # approach1_cross_site/
code_dir = os.path.dirname(approach_dir)    # code/
sys.path.insert(0, approach_dir)

from approach_1_utils import (
    load_config,
    load_and_preprocess_data,
    train_xgboost_approach1,
    train_nn_approach1,
    evaluate_model,
    save_model_artifacts
)

warnings.filterwarnings('ignore')


def main():
    """Main training function for Approach 1 Stage 1"""
    print("=" * 80)
    print("FLAME-ICU Approach 1 Stage 1: Cross-Site Model Training")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load configuration
    config_path = os.path.join(approach_dir, 'approach_1_config.json')
    config = load_config(config_path)

    print(f"Approach: {config['approach_name']}")
    print(f"Stage: {config['stage']}")
    print()

    # Set random seeds for reproducibility
    np.random.seed(config['random_seeds']['global'])

    try:
        # Step 1: Load and preprocess data
        print("📊 STEP 1: Loading and preprocessing data")
        print("-" * 50)
        splits, feature_names = load_and_preprocess_data(config)
        print(f"✅ Data loaded successfully with {len(feature_names)} features")
        print()

        # Step 2: Train XGBoost model
        print("🌳 STEP 2: Training XGBoost model")
        print("-" * 50)
        xgb_model, xgb_evals = train_xgboost_approach1(splits, config, feature_names)
        print("✅ XGBoost training completed")
        print()

        # Step 3: Train Neural Network model
        print("🧠 STEP 3: Training Neural Network model")
        print("-" * 50)
        nn_model, nn_history = train_nn_approach1(splits, config, feature_names)
        print("✅ Neural Network training completed")
        print()

        # Step 4: Evaluate models
        print("📈 STEP 4: Evaluating models")
        print("-" * 50)

        print("Evaluating XGBoost model...")
        xgb_results = evaluate_model(xgb_model, splits, config, model_type='xgboost')

        print("\nEvaluating Neural Network model...")
        nn_results = evaluate_model(nn_model, splits, config, model_type='nn')
        print()

        # Step 5: Save model artifacts
        print("💾 STEP 5: Saving model artifacts")
        print("-" * 50)

        # Save XGBoost model (no transformer needed since data is pre-scaled)
        xgb_model_dir = save_model_artifacts(
            xgb_model, None, feature_names, config, 'xgboost', xgb_results
        )

        # Save Neural Network model (no transformer needed since data is pre-scaled)
        nn_model_dir = save_model_artifacts(
            nn_model, None, feature_names, config, 'nn', nn_results
        )
        print()

        # Step 6: Create summary report
        print("📋 STEP 6: Creating summary report")
        print("-" * 50)
        create_summary_report(config, xgb_results, nn_results, feature_names)
        print()

        print("=" * 80)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("Model Performance Summary:")
        print(f"XGBoost Val AUC: {xgb_results['val']['metrics']['roc_auc']:.4f}")
        print(f"Neural Network Val AUC: {nn_results['val']['metrics']['roc_auc']:.4f}")
        print()
        print("Trained models saved and ready for Stage 2 distribution!")
        print(f"XGBoost model: {xgb_model_dir}")
        print(f"Neural Network model: {nn_model_dir}")

    except Exception as e:
        print(f"❌ ERROR: Training failed with error: {str(e)}")
        raise


def create_summary_report(config, xgb_results, nn_results, feature_names):
    """Create a comprehensive summary report"""

    # Get output directory
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, config['output_paths']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)

    # Create summary data
    summary = {
        'approach': config['approach_name'],
        'stage': config['stage'],
        'training_timestamp': datetime.now().isoformat(),
        'data_info': {
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'scaling_method': 'RobustScaler (applied in preprocessing)',
            'missing_value_handling': config['preprocessing']['handle_missing']
        },
        'model_performance': {
            'xgboost': {
                'val_auc': xgb_results['val']['metrics']['roc_auc'],
                'val_accuracy': xgb_results['val']['metrics']['accuracy'],
                'val_precision': xgb_results['val']['metrics']['precision'],
                'val_recall': xgb_results['val']['metrics']['recall'],
                'val_f1': xgb_results['val']['metrics']['f1_score']
            },
            'neural_network': {
                'val_auc': nn_results['val']['metrics']['roc_auc'],
                'val_accuracy': nn_results['val']['metrics']['accuracy'],
                'val_precision': nn_results['val']['metrics']['precision'],
                'val_recall': nn_results['val']['metrics']['recall'],
                'val_f1': nn_results['val']['metrics']['f1_score']
            }
        },
        'next_steps': {
            'stage_2_instructions': "Use trained models for cross-site testing",
            'model_locations': {
                'xgboost': f"{config['output_paths']['models_dir']}xgboost/",
                'neural_network': f"{config['output_paths']['models_dir']}nn/"
            }
        }
    }

    # Save summary
    summary_path = os.path.join(results_dir, 'approach_1_stage_1_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()