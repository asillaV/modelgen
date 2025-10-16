"""Training and prediction functions for RandomForest models."""

import logging
import time
from typing import Any, Literal, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from modelgen.ml.pipelines import create_preprocessing_pipeline

logger = logging.getLogger(__name__)

TaskType = Literal["regression", "classification"]


def create_model(
    task: TaskType,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: int = 42,
) -> Any:
    """Create a RandomForest model for the specified task.

    Args:
        task: Task type ('regression' or 'classification').
        n_estimators: Number of trees in the forest (50-500).
        max_depth: Maximum depth of trees (None or 2-20).
        random_state: Random state for reproducibility.

    Returns:
        Configured RandomForest model (Regressor or Classifier).

    Raises:
        ValueError: If task is not 'regression' or 'classification'.

    Example:
        >>> model = create_model('regression', n_estimators=100)
    """
    if task == "regression":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        logger.info(
            f"Created RandomForestRegressor: n_estimators={n_estimators}, "
            f"max_depth={max_depth}"
        )
    elif task == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        logger.info(
            f"Created RandomForestClassifier: n_estimators={n_estimators}, "
            f"max_depth={max_depth}"
        )
    else:
        raise ValueError(f"Invalid task: {task}. Must be 'regression' or 'classification'")

    return model


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    task: TaskType,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, Any]]:
    """Train a RandomForest model with preprocessing pipeline.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        numeric_features: List of numeric feature column names.
        categorical_features: List of categorical feature column names.
        task: Task type ('regression' or 'classification').
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of trees (None for no limit).
        test_size: Fraction of data to use for testing (0.1-0.3).
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (trained_pipeline, X_train, X_test, y_train, y_test, training_info).

    Example:
        >>> pipeline, X_train, X_test, y_train, y_test, info = train_model(
        ...     X, y, numeric_features, categorical_features, 'regression'
        ... )
    """
    logger.info(f"Starting training: task={task}, n_estimators={n_estimators}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(
        f"Data split: train={len(X_train)}, test={len(X_test)} "
        f"(test_size={test_size})"
    )

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        numeric_features, categorical_features
    )

    # Create model
    model = create_model(task, n_estimators, max_depth, random_state)

    # Create full pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Train model
    logger.info("Training model...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Collect training info
    training_info = {
        "task": task,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "test_size": test_size,
        "random_state": random_state,
        "training_time_seconds": training_time,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_features": X_train.shape[1],
    }

    return pipeline, X_train, X_test, y_train, y_test, training_info


def predict(pipeline: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Make predictions using a trained pipeline.

    Args:
        pipeline: Trained Pipeline object.
        X: Feature DataFrame.

    Returns:
        Predictions as a Series.

    Example:
        >>> y_pred = predict(pipeline, X_test)
    """
    logger.info(f"Making predictions for {len(X)} samples")
    predictions = pipeline.predict(X)
    return pd.Series(predictions, index=X.index)


def get_feature_importance(
    pipeline: Pipeline, feature_names: list[str]
) -> pd.DataFrame:
    """Extract feature importance from a trained RandomForest pipeline.

    Args:
        pipeline: Trained Pipeline object containing a RandomForest model.
        feature_names: Original feature names (before preprocessing).

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted by importance.

    Example:
        >>> importance_df = get_feature_importance(pipeline, feature_names)
    """
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        logger.warning("Model does not have feature_importances_ attribute")
        return pd.DataFrame(columns=["feature", "importance"])

    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        transformed_feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback if get_feature_names_out is not available
        transformed_feature_names = [
            f"feature_{i}" for i in range(len(model.feature_importances_))
        ]

    # Create importance DataFrame
    importance_df = pd.DataFrame(
        {
            "feature": transformed_feature_names,
            "importance": model.feature_importances_,
        }
    )

    # Sort by importance
    importance_df = importance_df.sort_values("importance", ascending=False)

    logger.info(f"Extracted feature importance for {len(importance_df)} features")

    return importance_df
