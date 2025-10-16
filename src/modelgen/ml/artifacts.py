"""Artifact management for saving and loading models, predictions, and reports."""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def save_model(pipeline: Pipeline, output_dir: Path, filename: str = "model.pkl") -> Path:
    """Save trained pipeline to disk using joblib.

    Args:
        pipeline: Trained Pipeline object.
        output_dir: Directory to save the model.
        filename: Filename for the model (default: 'model.pkl').

    Returns:
        Path to the saved model file.

    Example:
        >>> model_path = save_model(pipeline, Path("output"))
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / filename

    joblib.dump(pipeline, model_path)
    logger.info(f"Model saved to {model_path}")

    return model_path


def load_model(model_path: Path) -> Pipeline:
    """Load a trained pipeline from disk.

    Args:
        model_path: Path to the saved model file.

    Returns:
        Loaded Pipeline object.

    Raises:
        FileNotFoundError: If the model file does not exist.

    Example:
        >>> pipeline = load_model(Path("output/model.pkl"))
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipeline = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")

    return pipeline


def save_predictions(
    predictions: pd.Series,
    output_dir: Path,
    filename: str = "predictions.csv",
    include_index: bool = True,
) -> Path:
    """Save predictions to CSV file.

    Args:
        predictions: Series of predictions.
        output_dir: Directory to save the predictions.
        filename: Filename for the predictions (default: 'predictions.csv').
        include_index: Whether to include the index in the CSV.

    Returns:
        Path to the saved predictions file.

    Example:
        >>> pred_path = save_predictions(y_pred, Path("output"))
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / filename

    predictions.to_csv(pred_path, index=include_index, header=["prediction"])
    logger.info(f"Predictions saved to {pred_path} ({len(predictions)} rows)")

    return pred_path


def save_report(
    report_data: dict[str, Any],
    output_dir: Path,
    filename: str = "report.json",
) -> Path:
    """Save training/evaluation report as JSON.

    Args:
        report_data: Dictionary containing report information.
        output_dir: Directory to save the report.
        filename: Filename for the report (default: 'report.json').

    Returns:
        Path to the saved report file.

    Example:
        >>> report_path = save_report(training_info, Path("output"))
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / filename

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, default=str)

    logger.info(f"Report saved to {report_path}")

    return report_path


def create_full_report(
    training_info: dict[str, Any],
    metrics: dict[str, Any],
    feature_columns: list[str],
    target_column: str,
) -> dict[str, Any]:
    """Create a comprehensive report combining training info and metrics.

    Args:
        training_info: Dictionary with training parameters and timing.
        metrics: Dictionary with evaluation metrics.
        feature_columns: List of feature column names used.
        target_column: Name of the target column.

    Returns:
        Combined report dictionary.

    Example:
        >>> report = create_full_report(training_info, metrics, features, target)
    """
    report = {
        "target": target_column,
        "features": feature_columns,
        "training": training_info,
        "metrics": metrics,
    }

    return report


def save_all_artifacts(
    pipeline: Pipeline,
    predictions: pd.Series,
    report_data: dict[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    """Save all training artifacts (model, predictions, report) to output directory.

    Args:
        pipeline: Trained Pipeline object.
        predictions: Series of predictions.
        report_data: Dictionary containing report information.
        output_dir: Directory to save all artifacts.

    Returns:
        Dictionary mapping artifact names to their file paths.

    Example:
        >>> paths = save_all_artifacts(pipeline, y_pred, report, Path("output"))
        >>> print(paths["model"])
    """
    logger.info(f"Saving all artifacts to {output_dir}")

    paths = {
        "model": save_model(pipeline, output_dir),
        "predictions": save_predictions(predictions, output_dir),
        "report": save_report(report_data, output_dir),
    }

    logger.info("All artifacts saved successfully")

    return paths
