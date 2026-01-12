"""
Evaluation metrics for FLAIR benchmark tasks.

Task-specific metrics:
- Task 5 (ICU LOS): MSE, RMSE, MAE, R2
- Tasks 6, 7 (Classification): AUROC, AUPRC, F1, accuracy, ICI, DCA
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    brier_score_loss, confusion_matrix, roc_curve
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
            'n_negative': int(tn + fp),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
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

        # Store count metrics directly as integers (not as CI objects)
        count_metrics = ['tp', 'tn', 'fp', 'fn', 'n_positive', 'n_negative']
        base_metrics = self.evaluate(y_true, y_pred, y_pred_proba, threshold)
        for metric in count_metrics:
            if metric in base_metrics:
                ci_results[metric] = base_metrics[metric]

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
            if metric in ['auroc', 'auprc', 'accuracy', 'precision', 'recall', 'f1', 'specificity', 'npv', 'r2', 'ici']:
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value:.2f}")
        else:
            print(f"  {metric}: {value}")


# =============================================================================
# ICI, DCA, and ROC Curve Functions
# =============================================================================

def compute_ici(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """
    Compute Integrated Calibration Index using binned calibration.

    ICI = mean absolute difference between observed probability and
    predicted probability across calibration bins.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for calibration (default: 10)

    Returns:
        Dictionary with 'ici' value and 'calibration_curve' data
    """
    from sklearn.calibration import calibration_curve

    # Ensure arrays
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    try:
        # Compute binned calibration curve
        # fraction_of_positives = observed probability in each bin
        # mean_predicted_value = mean predicted probability in each bin
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
        )

        # Compute ICI: mean absolute calibration error across bins
        # Weight by bin size for more accurate ICI
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_weights = []

        for i in range(n_bins):
            # Count samples in this bin
            in_bin = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
            if i == n_bins - 1:  # Last bin includes right edge
                in_bin = (y_pred_proba >= bin_edges[i]) & (y_pred_proba <= bin_edges[i + 1])
            bin_weights.append(np.sum(in_bin))

        bin_weights = np.array(bin_weights)
        bin_weights = bin_weights / np.sum(bin_weights)  # Normalize to sum to 1

        # Weighted ICI
        calibration_errors = np.abs(fraction_of_positives - mean_predicted_value)
        ici = float(np.sum(calibration_errors * bin_weights[:len(calibration_errors)]))

        return {
            'ici': ici,
            'calibration_curve': {
                'predicted': mean_predicted_value.tolist(),
                'observed_smoothed': fraction_of_positives.tolist()
            }
        }

    except Exception as e:
        print(f"  Warning: Calibration curve computation failed ({e})")
        # Return empty calibration data
        return {
            'ici': 0.0,
            'calibration_curve': {
                'predicted': [],
                'observed_smoothed': []
            }
        }


def compute_dca(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute Decision Curve Analysis.

    Net Benefit = (TP/n) - (FP/n) * (threshold / (1 - threshold))

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        thresholds: Array of threshold probabilities (default: 0.01 to 0.99)

    Returns:
        Dictionary with threshold array and net benefit arrays
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    n = len(y_true)
    prevalence = np.mean(y_true)

    net_benefit_model = []
    net_benefit_all = []
    net_benefit_none = []

    for threshold in thresholds:
        # Model predictions at this threshold
        y_pred = (y_pred_proba >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # Net benefit for model
        if threshold < 1.0:
            nb_model = (tp / n) - (fp / n) * (threshold / (1 - threshold))
        else:
            nb_model = 0.0
        net_benefit_model.append(float(nb_model))

        # Net benefit for treat all
        if threshold < 1.0:
            nb_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        else:
            nb_all = 0.0
        net_benefit_all.append(float(max(nb_all, 0)))  # Cap at 0 for clinical relevance

        # Net benefit for treat none (always 0)
        net_benefit_none.append(0.0)

    return {
        'thresholds': thresholds.tolist(),
        'net_benefit_model': net_benefit_model,
        'net_benefit_all': net_benefit_all,
        'net_benefit_none': net_benefit_none
    }


def compute_roc_curve_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    max_points: int = 100
) -> Dict[str, Any]:
    """
    Extract ROC curve data for storage and plotting.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        max_points: Maximum number of points to store (downsampled if needed)

    Returns:
        Dictionary with fpr, tpr, thresholds arrays and auc
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    # Downsample if too many points
    if len(fpr) > max_points:
        indices = np.linspace(0, len(fpr) - 1, max_points, dtype=int)
        fpr = fpr[indices]
        tpr = tpr[indices]
        thresholds = thresholds[indices]

    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'auc': float(auc)
    }


def plot_roc_curve(
    roc_data: Dict[str, Any],
    output_path: str,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 8)
) -> str:
    """
    Create and save ROC curve plot.

    Args:
        roc_data: Dictionary from compute_roc_curve_data()
        output_path: Path to save PNG file
        title: Plot title
        figsize: Figure size

    Returns:
        Path to saved file
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(roc_data['fpr'], roc_data['tpr'],
            color='blue', lw=2,
            label=f"ROC (AUC = {roc_data['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_dca_curve(
    dca_data: Dict[str, Any],
    output_path: str,
    title: str = "Decision Curve Analysis",
    figsize: Tuple[int, int] = (10, 8)
) -> str:
    """
    Create and save DCA plot.

    Args:
        dca_data: Dictionary from compute_dca()
        output_path: Path to save PNG file
        title: Plot title
        figsize: Figure size

    Returns:
        Path to saved file
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    thresholds = np.array(dca_data['thresholds']) * 100  # Convert to percentage

    ax.plot(thresholds, dca_data['net_benefit_model'],
            color='blue', lw=2, label='Model')
    ax.plot(thresholds, dca_data['net_benefit_all'],
            color='gray', lw=1.5, linestyle='--', label='Treat All')
    ax.plot(thresholds, dca_data['net_benefit_none'],
            color='black', lw=1.5, linestyle=':', label='Treat None')

    ax.set_xlim([0, 100])
    # Set y limits: slightly below 0 to max positive benefit
    max_nb = max(max(dca_data['net_benefit_model']), max(dca_data['net_benefit_all']))
    y_margin = max_nb * 0.1 if max_nb > 0 else 0.05
    ax.set_ylim([-0.02, max(0.1, max_nb + y_margin)])

    ax.set_xlabel('Threshold Probability (%)', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: str,
    title: str = "Calibration Curve",
    n_bins: int = 10,
    figsize: Tuple[int, int] = (10, 8)
) -> str:
    """
    Create and save calibration curve plot with binned calibration.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        output_path: Path to save PNG file
        title: Plot title
        n_bins: Number of bins for histogram calibration
        figsize: Figure size

    Returns:
        Path to saved file
    """
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()

    # Binned calibration
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )

    fig, ax = plt.subplots(figsize=figsize)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect Calibration')

    # Binned calibration points
    ax.plot(mean_predicted_value, fraction_of_positives, 's-',
            color='blue', markersize=8, lw=2, label=f'Model (n={n_bins})')

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return output_path


def plot_predicted_vs_observed(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    title: str = "Predicted vs Observed",
    figsize: Tuple[int, int] = (8, 8),
    alpha: float = 0.3,
    loess_frac: float = 0.3
) -> str:
    """
    Create scatter plot of predicted vs observed values with LOESS trend line.

    For regression tasks (e.g., ICU LOS prediction).

    Args:
        y_true: True values
        y_pred: Predicted values
        output_path: Path to save PNG file
        title: Plot title
        figsize: Figure size
        alpha: Scatter point transparency
        loess_frac: LOESS smoothing fraction

    Returns:
        Path to saved file
    """
    import matplotlib.pyplot as plt
    from statsmodels.nonparametric.smoothers_lowess import lowess

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(y_pred, y_true, alpha=alpha, s=10, color='blue', label='Observations')

    # Perfect prediction line (diagonal)
    min_val = min(y_pred.min(), y_true.min())
    max_val = max(y_pred.max(), y_true.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Perfect Prediction')

    # LOESS smoothed curve
    sorted_idx = np.argsort(y_pred)
    y_pred_sorted = y_pred[sorted_idx]
    y_true_sorted = y_true[sorted_idx]

    loess_result = lowess(y_true_sorted, y_pred_sorted, frac=loess_frac, return_sorted=True)
    ax.plot(loess_result[:, 0], loess_result[:, 1], 'r-', lw=2, label='LOESS Trend')

    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Observed', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def compute_extended_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute all classification metrics including ICI, DCA, and ROC curve data.

    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary with all metrics and curve data
    """
    evaluator = TaskEvaluator('classification')
    y_pred_class = (y_pred_proba >= threshold).astype(int)

    # Base metrics
    results = evaluator.evaluate(y_true, y_pred_class, y_pred_proba, threshold)

    # Add ICI
    ici_data = compute_ici(y_true, y_pred_proba)
    results['ici'] = ici_data['ici']
    results['calibration_curve'] = ici_data['calibration_curve']

    # Add ROC curve data
    results['roc_curve'] = compute_roc_curve_data(y_true, y_pred_proba)

    # Add DCA data
    results['dca_curve'] = compute_dca(y_true, y_pred_proba)

    return results
