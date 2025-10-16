"""Visualization utilities for model predictions and feature importance."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_predictions_overlay(
    y_true: pd.Series,
    y_pred: pd.Series,
    x_axis: Optional[pd.Series] = None,
    title: str = "Predictions vs Actual",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot overlay of true vs predicted values.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        x_axis: Optional x-axis values (uses index if None).
        title: Plot title.
        ax: Optional matplotlib Axes to plot on.

    Returns:
        Matplotlib Axes object.

    Example:
        >>> ax = plot_predictions_overlay(y_test, y_pred)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    x = x_axis if x_axis is not None else y_true.index

    ax.plot(x, y_true, label="Actual", marker="o", linestyle="-", alpha=0.7)
    ax.plot(x, y_pred, label="Predicted", marker="x", linestyle="--", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("X-axis" if x_axis is not None else "Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_moving_average(
    y_true: pd.Series,
    y_pred: pd.Series,
    window: int = 5,
    x_axis: Optional[pd.Series] = None,
    title: str = "Moving Average Comparison",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot moving average of true vs predicted values.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        window: Window size for moving average.
        x_axis: Optional x-axis values (uses index if None).
        title: Plot title.
        ax: Optional matplotlib Axes to plot on.

    Returns:
        Matplotlib Axes object.

    Example:
        >>> ax = plot_moving_average(y_test, y_pred, window=10)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    x = x_axis if x_axis is not None else y_true.index

    y_true_ma = y_true.rolling(window=window, center=True).mean()
    y_pred_ma = y_pred.rolling(window=window, center=True).mean()

    ax.plot(x, y_true_ma, label=f"Actual (MA-{window})", linestyle="-", linewidth=2)
    ax.plot(
        x, y_pred_ma, label=f"Predicted (MA-{window})", linestyle="--", linewidth=2
    )
    ax.set_title(title)
    ax.set_xlabel("X-axis" if x_axis is not None else "Index")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return ax


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 10,
    title: str = "Feature Importance",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot feature importance bar chart.

    Args:
        importance_df: DataFrame with columns ['feature', 'importance'].
        top_n: Number of top features to display.
        title: Plot title.
        ax: Optional matplotlib Axes to plot on.

    Returns:
        Matplotlib Axes object.

    Example:
        >>> ax = plot_feature_importance(importance_df, top_n=15)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    top_features = importance_df.head(top_n).sort_values("importance", ascending=True)

    ax.barh(top_features["feature"], top_features["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(True, alpha=0.3, axis="x")

    return ax


def create_report_figure(
    y_true: pd.Series,
    y_pred: pd.Series,
    importance_df: Optional[pd.DataFrame] = None,
    x_axis: Optional[pd.Series] = None,
    window: int = 5,
    metrics_text: Optional[str] = None,
) -> plt.Figure:
    """Create a composite figure with multiple subplots for the report.

    Args:
        y_true: True target values.
        y_pred: Predicted target values.
        importance_df: Optional DataFrame with feature importance.
        x_axis: Optional x-axis values.
        window: Window size for moving average.
        metrics_text: Optional text with metrics to display.

    Returns:
        Matplotlib Figure object with subplots.

    Example:
        >>> fig = create_report_figure(y_test, y_pred, importance_df)
    """
    n_plots = 3 if importance_df is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))

    if n_plots == 1:
        axes = [axes]

    # Plot 1: Overlay
    title = "Predictions vs Actual"
    if metrics_text:
        title += f"\n{metrics_text}"
    plot_predictions_overlay(y_true, y_pred, x_axis, title=title, ax=axes[0])

    # Plot 2: Moving Average
    plot_moving_average(y_true, y_pred, window, x_axis, ax=axes[1])

    # Plot 3: Feature Importance (if available)
    if importance_df is not None and len(importance_df) > 0:
        plot_feature_importance(importance_df, top_n=10, ax=axes[2])

    plt.tight_layout()

    return fig


def save_report_figure(
    fig: plt.Figure, output_dir: Path, filename: str = "report.png"
) -> Path:
    """Save figure to file.

    Args:
        fig: Matplotlib Figure object.
        output_dir: Directory to save the figure.
        filename: Filename for the figure.

    Returns:
        Path to the saved figure file.

    Example:
        >>> fig_path = save_report_figure(fig, Path("output"))
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / filename

    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    logger.info(f"Report figure saved to {fig_path}")

    plt.close(fig)

    return fig_path
