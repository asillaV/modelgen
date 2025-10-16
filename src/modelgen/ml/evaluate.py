"""Evaluation metrics for regression and classification tasks."""

import logging
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Evaluate regression predictions.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.

    Returns:
        Dictionary with regression metrics (R2, MSE, MAE).

    Example:
        >>> metrics = evaluate_regression(y_test, y_pred)
        >>> print(f"R²: {metrics['r2']:.3f}")
    """
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    metrics = {
        "r2": r2,
        "mse": mse,
        "mae": mae,
        "rmse": mse**0.5,
    }

    logger.info(
        f"Regression metrics: R²={r2:.4f}, MSE={mse:.4f}, "
        f"MAE={mae:.4f}, RMSE={metrics['rmse']:.4f}"
    )

    return metrics


def evaluate_classification(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    """Evaluate classification predictions.

    Args:
        y_true: True target labels.
        y_pred: Predicted target labels.

    Returns:
        Dictionary with classification metrics (accuracy, F1, confusion matrix).

    Example:
        >>> metrics = evaluate_classification(y_test, y_pred)
        >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Convert confusion matrix to dictionary format
    labels = sorted(set(y_true) | set(y_pred))
    conf_matrix_dict = {}
    for i, true_label in enumerate(labels):
        conf_matrix_dict[str(true_label)] = {}
        for j, pred_label in enumerate(labels):
            conf_matrix_dict[str(true_label)][str(pred_label)] = int(
                conf_matrix[i, j]
            )

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "confusion_matrix": conf_matrix_dict,
        "n_classes": len(labels),
    }

    logger.info(
        f"Classification metrics: Accuracy={accuracy:.4f}, "
        f"F1-Macro={f1_macro:.4f}, Classes={len(labels)}"
    )

    return metrics


def evaluate_model(
    y_true: pd.Series, y_pred: pd.Series, task: str
) -> dict[str, Any]:
    """Evaluate model predictions based on task type.

    Args:
        y_true: True target values/labels.
        y_pred: Predicted target values/labels.
        task: Task type ('regression' or 'classification').

    Returns:
        Dictionary with task-appropriate metrics.

    Raises:
        ValueError: If task is not 'regression' or 'classification'.

    Example:
        >>> metrics = evaluate_model(y_test, y_pred, 'regression')
    """
    if task == "regression":
        return evaluate_regression(y_true, y_pred)
    elif task == "classification":
        return evaluate_classification(y_true, y_pred)
    else:
        raise ValueError(
            f"Invalid task: {task}. Must be 'regression' or 'classification'"
        )
