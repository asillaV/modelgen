"""Tests for training and evaluation."""

import pandas as pd

from modelgen.ml.evaluate import evaluate_classification, evaluate_regression
from modelgen.ml.train import create_model, train_model


def test_create_regression_model():
    """Test creating a regression model."""
    model = create_model("regression", n_estimators=10)
    assert model is not None
    assert model.n_estimators == 10


def test_create_classification_model():
    """Test creating a classification model."""
    model = create_model("classification", n_estimators=10)
    assert model is not None
    assert model.n_estimators == 10


def test_train_regression():
    """Test training a simple regression model."""
    # Create simple dataset
    X = pd.DataFrame({"x1": [1, 2, 3, 4, 5], "x2": [2, 4, 6, 8, 10]})
    y = pd.Series([3, 6, 9, 12, 15])

    pipeline, X_train, X_test, y_train, y_test, info = train_model(
        X, y, numeric_features=["x1", "x2"], categorical_features=[], task="regression", n_estimators=10
    )

    assert pipeline is not None
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert info["task"] == "regression"


def test_evaluate_regression():
    """Test regression evaluation metrics."""
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = pd.Series([1.1, 2.1, 2.9, 4.1])

    metrics = evaluate_regression(y_true, y_pred)

    assert "r2" in metrics
    assert "mse" in metrics
    assert "mae" in metrics
    assert metrics["r2"] > 0.9


def test_evaluate_classification():
    """Test classification evaluation metrics."""
    y_true = pd.Series(["A", "B", "A", "B"])
    y_pred = pd.Series(["A", "B", "A", "B"])

    metrics = evaluate_classification(y_true, y_pred)

    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert metrics["accuracy"] == 1.0
