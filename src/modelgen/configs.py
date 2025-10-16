"""Configuration dataclasses for model training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Attributes:
        train_path: Path to training CSV file.
        test_path: Optional path to test CSV file (if None, split from train).
        task: Task type ('regression' or 'classification').
        target: Name of target column.
        features: List of feature column names (if None, use all except target).
        x_axis: Optional column to use as x-axis in plots.
        n_estimators: Number of trees in RandomForest (50-500).
        max_depth: Maximum tree depth (None for no limit, or 2-20).
        test_size: Fraction of data for testing (0.1-0.3).
        random_state: Random seed for reproducibility.
        output_dir: Directory to save artifacts.
        ma_window: Moving average window size for plots.
    """

    train_path: Path
    task: Literal["regression", "classification"]
    target: str
    test_path: Optional[Path] = None
    features: Optional[list[str]] = None
    x_axis: Optional[str] = None
    n_estimators: int = 100
    max_depth: Optional[int] = None
    test_size: float = 0.2
    random_state: int = 42
    output_dir: Path = field(default_factory=lambda: Path("output"))
    ma_window: int = 5

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.n_estimators < 50 or self.n_estimators > 500:
            raise ValueError("n_estimators must be between 50 and 500")

        if self.max_depth is not None and (
            self.max_depth < 2 or self.max_depth > 20
        ):
            raise ValueError("max_depth must be None or between 2 and 20")

        if self.test_size < 0.1 or self.test_size > 0.3:
            raise ValueError("test_size must be between 0.1 and 0.3")

        if self.task not in ["regression", "classification"]:
            raise ValueError("task must be 'regression' or 'classification'")


@dataclass
class CSVLoadConfig:
    """Configuration for CSV loading.

    Attributes:
        sample_rows: Optional maximum number of rows to load.
        encoding: Optional encoding override (None for auto-detect).
        delimiter: Optional delimiter override (None for auto-detect).
        header: Optional header row index (0 or None).
    """

    sample_rows: Optional[int] = None
    encoding: Optional[str] = None
    delimiter: Optional[str] = None
    header: Optional[int] = 0
