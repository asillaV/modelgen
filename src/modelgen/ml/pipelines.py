"""Preprocessing pipelines for numeric and categorical features."""

import logging
from typing import Optional

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)


def create_preprocessing_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    max_categories: int = 32,
) -> ColumnTransformer:
    """Create a preprocessing pipeline with separate handling for numeric and categorical features.

    Numeric features:
        - Imputation with median strategy

    Categorical features:
        - Imputation with most_frequent strategy
        - OneHotEncoding with unknown handling and category limit

    Args:
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.
        max_categories: Maximum number of categories for OneHotEncoder.

    Returns:
        Configured ColumnTransformer pipeline.

    Example:
        >>> numeric_cols = ['x1', 'x2']
        >>> categorical_cols = ['category']
        >>> pipeline = create_preprocessing_pipeline(numeric_cols, categorical_cols)
    """
    transformers = []

    # Numeric pipeline
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_features))
        logger.info(f"Numeric pipeline: {len(numeric_features)} features")

    # Categorical pipeline
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        max_categories=max_categories,
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))
        logger.info(
            f"Categorical pipeline: {len(categorical_features)} features "
            f"(max_categories={max_categories})"
        )

    if not transformers:
        raise ValueError("No features provided for preprocessing pipeline")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return preprocessor


def prepare_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Prepare features by separating them into numeric and categorical types.

    Args:
        df: Input DataFrame.
        feature_columns: List of feature column names to use.
        numeric_columns: List of all numeric column names in the DataFrame.
        categorical_columns: List of all categorical column names in the DataFrame.

    Returns:
        Tuple of (features_df, numeric_features, categorical_features).

    Raises:
        ValueError: If any feature column is not found in the DataFrame.

    Example:
        >>> df = pd.DataFrame({'x1': [1, 2], 'x2': [3, 4], 'cat': ['A', 'B']})
        >>> features_df, num, cat = prepare_features(
        ...     df, ['x1', 'cat'], ['x1', 'x2'], ['cat']
        ... )
    """
    # Validate that all feature columns exist
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found in DataFrame: {missing}")

    # Extract features DataFrame
    X = df[feature_columns].copy()

    # Separate numeric and categorical features
    numeric_features = [col for col in feature_columns if col in numeric_columns]
    categorical_features = [col for col in feature_columns if col in categorical_columns]

    logger.info(
        f"Prepared {len(feature_columns)} features: "
        f"{len(numeric_features)} numeric, {len(categorical_features)} categorical"
    )

    return X, numeric_features, categorical_features


def prepare_target(
    df: pd.DataFrame, target_column: str, task: str = "regression"
) -> pd.Series:
    """Prepare target variable and validate it for the specified task.

    Args:
        df: Input DataFrame.
        target_column: Name of the target column.
        task: Task type ('regression' or 'classification').

    Returns:
        Target variable as a Series.

    Raises:
        ValueError: If target column is not found or has invalid type for the task.

    Example:
        >>> df = pd.DataFrame({'y': [1, 2, 3]})
        >>> y = prepare_target(df, 'y', task='regression')
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    y = df[target_column].copy()

    # Check for missing values in target
    if y.isna().any():
        missing_count = y.isna().sum()
        logger.warning(
            f"Target column '{target_column}' contains {missing_count} missing values. "
            "Rows with missing targets will be dropped."
        )
        y = y.dropna()

    # Validate target type for task
    if task == "regression":
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                f"Target column '{target_column}' must be numeric for regression task. "
                f"Found dtype: {y.dtype}"
            )
    elif task == "classification":
        unique_values = y.nunique()
        if unique_values < 2:
            raise ValueError(
                f"Target column '{target_column}' must have at least 2 unique values "
                f"for classification. Found: {unique_values}"
            )
        logger.info(f"Classification target has {unique_values} unique classes")

    logger.info(f"Prepared target '{target_column}' for {task} task")

    return y
