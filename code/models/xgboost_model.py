"""
XGBoost model wrapper for FLAIR benchmark.

Supports both classification (tasks 6, 7) and regression (task 5).
Reuses optimized parameters from old/approach1_cross_site/approach_1_config.json.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path


# Optimized XGBoost parameters (from old/approach1_cross_site/approach_1_config.json)
DEFAULT_PARAMS = {
    'eta': 0.06057610120076962,
    'max_depth': 8,
    'min_child_weight': 6,
    'max_delta_step': 0,
    'subsample': 0.9274884839441507,
    'colsample_bytree': 0.9225204370066138,
    'gamma': 0.1652835907230118,
    'reg_alpha': 8.404509395484988,
    'reg_lambda': 0.10842839237042663,
    'seed': 42,
    'num_rounds': 500,
    'early_stopping_rounds': 20,
    'use_class_weights': True
}


class XGBoostModel:
    """
    XGBoost model wrapper for FLAIR benchmark tasks.

    Supports:
    - Task 5 (ICU LOS): Regression
    - Task 6 (Hospital Mortality): Binary classification
    - Task 7 (ICU Readmission): Binary classification
    """

    def __init__(
        self,
        task_type: str = 'classification',
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize XGBoost model.

        Args:
            task_type: 'classification' or 'regression'
            params: Optional custom parameters (uses defaults if not provided)
        """
        self.task_type = task_type
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.evals_result = {}
        self.feature_names = None

    def _prepare_params(self, y_train: np.ndarray) -> Dict[str, Any]:
        """Prepare XGBoost parameters based on task type."""
        params = self.params.copy()

        # Remove non-XGBoost params
        num_rounds = params.pop('num_rounds', 500)
        early_stopping_rounds = params.pop('early_stopping_rounds', 20)
        use_class_weights = params.pop('use_class_weights', True)

        if self.task_type == 'classification':
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'

            # Calculate class weights for imbalanced data
            if use_class_weights:
                class_counts = np.bincount(y_train.astype(int))
                if len(class_counts) > 1:
                    scale_pos_weight = class_counts[0] / max(1, class_counts[1])
                    params['scale_pos_weight'] = scale_pos_weight
                    print(f"  Using scale_pos_weight: {scale_pos_weight:.2f}")
        else:
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'rmse'

        return params, num_rounds, early_stopping_rounds

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'XGBoostModel':
        """
        Train the XGBoost model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, enables early stopping)
            y_val: Validation labels (optional)
            verbose: Print training progress

        Returns:
            Self for chaining
        """
        print(f"Training XGBoost ({self.task_type})...")
        print(f"  Train shape: {X_train.shape}")

        self.feature_names = list(X_train.columns)
        params, num_rounds, early_stopping_rounds = self._prepare_params(y_train)

        # XGBoost handles missing values natively
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, 'val'))
            print(f"  Val shape: {X_val.shape}")

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
            evals_result=self.evals_result,
            verbose_eval=50 if verbose else False
        )

        print(f"  Training complete. Best iteration: {self.model.best_iteration}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions.

        Args:
            X: Features DataFrame

        Returns:
            Predictions (probabilities for classification, values for regression)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions (classification only).

        Args:
            X: Features DataFrame

        Returns:
            Predicted probabilities for positive class
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        return self.predict(X)

    def predict_class(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Get class predictions (classification only).

        Args:
            X: Features DataFrame
            threshold: Classification threshold

        Returns:
            Binary class predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            importance_type: 'gain', 'weight', or 'cover'

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        importance = self.model.get_score(importance_type=importance_type)
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ]).sort_values('importance', ascending=False)

        return df

    def save(self, path: str) -> None:
        """
        Save model to JSON file.

        Args:
            path: Output file path
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        print(f"Model saved to: {path}")

    def load(self, path: str) -> 'XGBoostModel':
        """
        Load model from JSON file.

        Args:
            path: Input file path

        Returns:
            Self for chaining
        """
        self.model = xgb.Booster()
        self.model.load_model(path)
        print(f"Model loaded from: {path}")
        return self
