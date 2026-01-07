"""
Evaluation metrics for FLAIR benchmark tasks.

Task-specific metrics:
- Task 5 (ICU LOS): MSE, RMSE, MAE, R2
- Tasks 6, 7 (Classification): AUROC, AUPRC, F1, accuracy
"""

import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    brier_score_loss, confusion_matrix
)


class TaskEvaluator:
    """
    Evaluate model predictions based on task type.

    Supports:
    - Task 5 (ICU LOS): Regression metrics
    - Tasks 6, 7 (Classification): Binary classification metrics
    """

    def __init__(self, task_type: str):
        """
        Initialize evaluator.

        Args:
            task_type: 'regression' or 'classification'
        """
        self.task_type = task_type

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Evaluate predictions and return metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted values (for regression) or classes (for classification)
            y_pred_proba: Predicted probabilities (classification only)
            threshold: Classification threshold (for computing class predictions from proba)

        Returns:
            Dictionary of metric names to values
        """
        if self.task_type == 'regression':
            return self._evaluate_regression(y_true, y_pred)
        else:
            # If y_pred_proba provided, compute class predictions
            if y_pred_proba is not None:
                y_pred_class = (y_pred_proba >= threshold).astype(int)
            else:
                y_pred_class = y_pred
                y_pred_proba = y_pred  # Assume y_pred is probabilities

            return self._evaluate_classification(y_true, y_pred_class, y_pred_proba)

    def _evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Regression metrics for Task 5 (ICU LOS).

        Returns:
            MSE, RMSE, MAE, R2
        """
        mse = mean_squared_error(y_true, y_pred)

        return {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred))
        }

    def _evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Classification metrics for Tasks 6, 7.

        Returns:
            AUROC, AUPRC, accuracy, precision, recall, F1, specificity, NPV, Brier score
        """
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        results = {
            'auroc': float(roc_auc_score(y_true, y_pred_proba)),
            'auprc': float(average_precision_score(y_true, y_pred_proba)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0)),
            'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            'npv': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
            'brier_score': float(brier_score_loss(y_true, y_pred_proba)),
            'n_positive': int(tp + fn),
            'n_negative': int(tn + fp)
        }

        return results

    def compute_bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        n_iterations: int = 1000,
        confidence: float = 0.95,
        threshold: float = 0.5,
        random_seed: int = 42
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute bootstrap confidence intervals for metrics.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted values or classes
            y_pred_proba: Predicted probabilities (classification only)
            n_iterations: Number of bootstrap iterations
            confidence: Confidence level (e.g., 0.95 for 95% CI)
            threshold: Classification threshold
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with metric names mapping to {mean, ci_lower, ci_upper}
        """
        np.random.seed(random_seed)

        bootstrap_results = {}
        n_samples = len(y_true)

        for i in range(n_iterations):
            # Stratified bootstrap for classification
            if self.task_type == 'classification':
                pos_idx = np.where(y_true == 1)[0]
                neg_idx = np.where(y_true == 0)[0]

                pos_sample = np.random.choice(pos_idx, size=len(pos_idx), replace=True)
                neg_sample = np.random.choice(neg_idx, size=len(neg_idx), replace=True)

                boot_idx = np.concatenate([pos_sample, neg_sample])
            else:
                boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)

            y_true_boot = y_true[boot_idx]
            y_pred_boot = y_pred[boot_idx]
            y_proba_boot = y_pred_proba[boot_idx] if y_pred_proba is not None else None

            try:
                metrics = self.evaluate(y_true_boot, y_pred_boot, y_proba_boot, threshold)
                for metric, value in metrics.items():
                    if metric not in bootstrap_results:
                        bootstrap_results[metric] = []
                    bootstrap_results[metric].append(value)
            except Exception:
                continue

        # Calculate CIs
        alpha = 1 - confidence
        ci_results = {}

        for metric, values in bootstrap_results.items():
            if values:
                ci_results[metric] = {
                    'mean': float(np.mean(values)),
                    'ci_lower': float(np.percentile(values, alpha / 2 * 100)),
                    'ci_upper': float(np.percentile(values, (1 - alpha / 2) * 100))
                }

        return ci_results


def evaluate_task5(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Convenience function for Task 5 (ICU LOS) evaluation.

    Args:
        y_true: True ICU LOS values in hours
        y_pred: Predicted ICU LOS values in hours

    Returns:
        Dictionary with MSE, RMSE, MAE, R2
    """
    evaluator = TaskEvaluator('regression')
    return evaluator.evaluate(y_true, y_pred)


def evaluate_task6(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Convenience function for Task 6 (Hospital Mortality) evaluation.

    Args:
        y_true: True mortality labels (0/1)
        y_pred_proba: Predicted mortality probabilities
        threshold: Classification threshold

    Returns:
        Dictionary with classification metrics
    """
    evaluator = TaskEvaluator('classification')
    y_pred_class = (y_pred_proba >= threshold).astype(int)
    return evaluator.evaluate(y_true, y_pred_class, y_pred_proba, threshold)


def evaluate_task7(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Convenience function for Task 7 (ICU Readmission) evaluation.

    Args:
        y_true: True readmission labels (0/1)
        y_pred_proba: Predicted readmission probabilities
        threshold: Classification threshold

    Returns:
        Dictionary with classification metrics
    """
    evaluator = TaskEvaluator('classification')
    y_pred_class = (y_pred_proba >= threshold).astype(int)
    return evaluator.evaluate(y_true, y_pred_class, y_pred_proba, threshold)


def print_metrics(metrics: Dict[str, float], title: str = "Metrics") -> None:
    """
    Pretty print metrics.

    Args:
        metrics: Dictionary of metric names to values
        title: Title to display
    """
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")

    for metric, value in metrics.items():
        if isinstance(value, float):
            if metric in ['auroc', 'auprc', 'accuracy', 'precision', 'recall', 'f1', 'specificity', 'npv', 'r2']:
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")
