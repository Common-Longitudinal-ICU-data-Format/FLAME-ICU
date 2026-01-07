"""
ElasticNet model wrapper for FLAIR benchmark.

Uses sklearn's ElasticNet for regression and LogisticRegression with elastic net for classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib


# Default ElasticNet parameters
DEFAULT_PARAMS = {
    'alpha': 1.0,
    'l1_ratio': 0.5,
    'max_iter': 1000,
    'tol': 0.0001,
    'random_state': 42
}


class ElasticNetModel:
    """
    ElasticNet model wrapper for FLAIR benchmark tasks.

    Uses:
    - sklearn ElasticNet for regression (Task 5: ICU LOS)
    - sklearn LogisticRegression with elastic net penalty for classification (Tasks 6, 7)
    """

    def __init__(
        self,
        task_type: str = 'classification',
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ElasticNet model.

        Args:
            task_type: 'classification' or 'regression'
            params: Optional custom parameters (uses defaults if not provided)
        """
        self.task_type = task_type
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.feature_medians = None

    def _create_model(self):
        """Create the appropriate sklearn model."""
        params = self.params.copy()

        if self.task_type == 'classification':
            # LogisticRegression with elastic net penalty (requires saga solver)
            return LogisticRegression(
                penalty='elasticnet',
                solver='saga',
                l1_ratio=params.get('l1_ratio', 0.5),
                C=1.0 / params.get('alpha', 1.0),  # C = 1/alpha
                max_iter=params.get('max_iter', 1000),
                tol=params.get('tol', 0.0001),
                random_state=params.get('random_state', 42),
                class_weight='balanced',
                n_jobs=-1
            )
        else:
            # Standard ElasticNet for regression
            return ElasticNet(
                alpha=params.get('alpha', 1.0),
                l1_ratio=params.get('l1_ratio', 0.5),
                max_iter=params.get('max_iter', 1000),
                tol=params.get('tol', 0.0001),
                random_state=params.get('random_state', 42)
            )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True
    ) -> 'ElasticNetModel':
        """
        Train the ElasticNet model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used, for API compatibility)
            y_val: Validation labels (not used, for API compatibility)
            verbose: Print training progress

        Returns:
            Self for chaining
        """
        print(f"Training ElasticNet ({self.task_type})...")
        print(f"  Train shape: {X_train.shape}")

        self.feature_names = list(X_train.columns)

        # Store medians for imputation
        self.feature_medians = X_train.median()

        # Impute missing values with median (ElasticNet can't handle NaN)
        X_train_filled = X_train.fillna(self.feature_medians)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_filled)

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)

        if verbose:
            if self.task_type == 'classification':
                # Get coefficients
                n_nonzero = np.sum(self.model.coef_ != 0)
                print(f"  Non-zero coefficients: {n_nonzero}/{len(self.model.coef_.flatten())}")
            else:
                n_nonzero = np.sum(self.model.coef_ != 0)
                print(f"  Non-zero coefficients: {n_nonzero}/{len(self.model.coef_)}")

        print("  Training complete.")
        return self

    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """Preprocess features (impute and scale)."""
        X_filled = X.fillna(self.feature_medians)
        X_scaled = self.scaler.transform(X_filled)
        return X_scaled

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get predictions.

        Args:
            X: Features DataFrame

        Returns:
            Predictions (values for regression, classes for classification)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self._preprocess(X)
        return self.model.predict(X_scaled)

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

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        X_scaled = self._preprocess(X)
        return self.model.predict_proba(X_scaled)[:, 1]

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

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (coefficient magnitudes).

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if self.task_type == 'classification':
            coef = self.model.coef_.flatten()
        else:
            coef = self.model.coef_

        df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coef,
            'importance': np.abs(coef)
        }).sort_values('importance', ascending=False)

        return df

    def save(self, path: str) -> None:
        """
        Save model, scaler, and medians to file.

        Args:
            path: Output file path
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_medians': self.feature_medians,
            'task_type': self.task_type,
            'params': self.params
        }
        joblib.dump(save_dict, path)
        print(f"Model saved to: {path}")

    def load(self, path: str) -> 'ElasticNetModel':
        """
        Load model, scaler, and medians from file.

        Args:
            path: Input file path

        Returns:
            Self for chaining
        """
        loaded = joblib.load(path)
        self.model = loaded['model']
        self.scaler = loaded['scaler']
        self.feature_names = loaded['feature_names']
        self.feature_medians = loaded['feature_medians']
        self.task_type = loaded['task_type']
        self.params = loaded['params']
        print(f"Model loaded from: {path}")
        return self
