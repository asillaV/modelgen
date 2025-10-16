"""Column type inference and schema utilities."""

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ColumnSchema:
    """Schema information for DataFrame columns.

    Attributes:
        numeric_columns: List of numeric column names.
        categorical_columns: List of categorical column names.
        datetime_columns: List of datetime column names.
        all_columns: List of all column names.
    """

    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    all_columns: list[str]


def infer_column_types(
    df: pd.DataFrame, datetime_threshold: float = 0.8
) -> ColumnSchema:
    """Infer column types from a DataFrame.

    Categorizes columns as numeric, categorical, or datetime based on their dtype
    and content.

    Args:
        df: Input DataFrame.
        datetime_threshold: Threshold for datetime detection (0-1). If a column
            can be converted to datetime with success rate above this threshold,
            it's classified as datetime.

    Returns:
        ColumnSchema object with categorized column lists.

    Example:
        >>> df = pd.DataFrame({'x': [1, 2, 3], 'label': ['A', 'B', 'A']})
        >>> schema = infer_column_types(df)
        >>> print(schema.numeric_columns)
        ['x']
        >>> print(schema.categorical_columns)
        ['label']
    """
    numeric_cols = []
    categorical_cols = []
    datetime_cols = []

    for col in df.columns:
        dtype = df[col].dtype

        # Check for datetime types
        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)
            logger.debug(f"Column '{col}' classified as datetime (dtype: {dtype})")
            continue

        # Check for numeric types
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)
            logger.debug(f"Column '{col}' classified as numeric (dtype: {dtype})")
            continue

        # Try to convert to datetime for string columns
        if pd.api.types.is_string_dtype(dtype) or dtype == object:
            # Attempt datetime conversion
            try:
                converted = pd.to_datetime(df[col], errors="coerce")
                success_rate = converted.notna().sum() / len(df[col])
                if success_rate >= datetime_threshold:
                    datetime_cols.append(col)
                    logger.debug(
                        f"Column '{col}' classified as datetime (success rate: {success_rate:.2f})"
                    )
                    continue
            except Exception:
                pass

            # Classify as categorical
            categorical_cols.append(col)
            logger.debug(f"Column '{col}' classified as categorical (dtype: {dtype})")
            continue

        # Default to categorical
        categorical_cols.append(col)
        logger.debug(f"Column '{col}' classified as categorical (default)")

    schema = ColumnSchema(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        all_columns=list(df.columns),
    )

    logger.info(
        f"Schema inference: {len(numeric_cols)} numeric, "
        f"{len(categorical_cols)} categorical, {len(datetime_cols)} datetime"
    )

    return schema


def validate_columns(
    df: pd.DataFrame, required_columns: list[str], context: str = ""
) -> None:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: Input DataFrame.
        required_columns: List of column names that must be present.
        context: Optional context string for error messages.

    Raises:
        ValueError: If any required column is missing.

    Example:
        >>> df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
        >>> validate_columns(df, ['x', 'y'], context="training")
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        ctx = f" ({context})" if context else ""
        raise ValueError(f"Missing required columns{ctx}: {missing}")


def select_x_axis(
    df: pd.DataFrame, x_axis: Optional[str], schema: ColumnSchema
) -> pd.Series:
    """Select or create an x-axis for plotting.

    If x_axis is provided and exists, use it. Otherwise, use the DataFrame index.

    Args:
        df: Input DataFrame.
        x_axis: Optional column name to use as x-axis.
        schema: ColumnSchema object for type checking.

    Returns:
        Series to use as x-axis for plotting.

    Raises:
        ValueError: If the specified x_axis column does not exist.

    Example:
        >>> df = pd.DataFrame({'time': [1, 2, 3], 'value': [10, 20, 30]})
        >>> schema = infer_column_types(df)
        >>> x = select_x_axis(df, 'time', schema)
    """
    if x_axis is None:
        logger.info("No x_axis specified, using DataFrame index")
        return pd.Series(df.index, index=df.index)

    if x_axis not in df.columns:
        raise ValueError(
            f"x_axis column '{x_axis}' not found. Available columns: {list(df.columns)}"
        )

    logger.info(f"Using column '{x_axis}' as x-axis")
    return df[x_axis]


def get_categorical_unique_counts(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """Get unique value counts for categorical columns.

    Args:
        df: Input DataFrame.
        columns: List of categorical column names.

    Returns:
        Dictionary mapping column names to their unique value counts.

    Example:
        >>> df = pd.DataFrame({'label': ['A', 'B', 'A', 'C']})
        >>> counts = get_categorical_unique_counts(df, ['label'])
        >>> counts['label']
        3
    """
    return {col: df[col].nunique() for col in columns if col in df.columns}
