"""Command-line interface for modelgen."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from modelgen import logging_setup
from modelgen.io.loaders import load_csv
from modelgen.io.schema import infer_column_types, select_x_axis
from modelgen.ml.artifacts import create_full_report, save_all_artifacts
from modelgen.ml.evaluate import evaluate_model
from modelgen.ml.pipelines import prepare_features, prepare_target
from modelgen.ml.train import get_feature_importance, predict, train_model
from modelgen.viz.plots import create_report_figure, save_report_figure


def train_command(args: argparse.Namespace) -> int:
    """Execute the train command.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    try:
        # Setup logging
        log_file = args.out / "train.log" if args.out else None
        logging_setup.setup_logging(level="INFO", log_file=log_file)
        logger = logging_setup.get_logger(__name__)

        logger.info("Starting training workflow")
        logger.info(f"Task: {args.task}, Target: {args.target}")

        # Load training data
        logger.info(f"Loading training data from {args.train}")
        df_train, load_report = load_csv(args.train)
        logger.info(f"Training data loaded: {load_report.shape}")

        # Infer column types
        schema = infer_column_types(df_train)

        # Determine feature columns
        if args.features:
            feature_columns = args.features
        else:
            # Use all columns except target
            feature_columns = [col for col in df_train.columns if col != args.target]

        logger.info(f"Using {len(feature_columns)} features")

        # Prepare features and target
        X, numeric_features, categorical_features = prepare_features(
            df_train, feature_columns, schema.numeric_columns, schema.categorical_columns
        )
        y = prepare_target(df_train, args.target, task=args.task)

        # Handle mismatched indices (due to missing values in target)
        X = X.loc[y.index]

        # Train model
        logger.info("Training model...")
        pipeline, X_train, X_test, y_train, y_test, training_info = train_model(
            X,
            y,
            numeric_features,
            categorical_features,
            task=args.task,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            test_size=args.test_size,
        )

        # Make predictions on test set
        y_pred = predict(pipeline, X_test)

        # Evaluate model
        logger.info("Evaluating model...")
        metrics = evaluate_model(y_test, y_pred, task=args.task)

        # Create report
        report_data = create_full_report(
            training_info, metrics, feature_columns, args.target
        )

        # Get feature importance
        importance_df = get_feature_importance(pipeline, feature_columns)

        # Create visualizations
        if args.task == "regression":
            logger.info("Creating visualizations...")
            x_axis_data = select_x_axis(df_train.loc[X_test.index], args.x_axis, schema)

            # Format metrics for plot title
            metrics_text = f"R²: {metrics['r2']:.3f}, MSE: {metrics['mse']:.2f}"

            fig = create_report_figure(
                y_test,
                y_pred,
                importance_df,
                x_axis=x_axis_data,
                window=args.ma_window,
                metrics_text=metrics_text,
            )
            fig_path = save_report_figure(fig, args.out)
            logger.info(f"Report figure saved to {fig_path}")

        # Save artifacts
        logger.info("Saving artifacts...")
        artifact_paths = save_all_artifacts(pipeline, y_pred, report_data, args.out)

        logger.info("Training completed successfully")
        logger.info(f"Model saved to {artifact_paths['model']}")
        logger.info(f"Report saved to {artifact_paths['report']}")

        # Print metrics to console
        print("\nTraining Results:")
        print("=" * 50)
        if args.task == "regression":
            print(f"R² Score: {metrics['r2']:.4f}")
            print(f"MSE: {metrics['mse']:.4f}")
            print(f"MAE: {metrics['mae']:.4f}")
            print(f"RMSE: {metrics['rmse']:.4f}")
        else:
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1-Macro: {metrics['f1_macro']:.4f}")
            print(f"Number of classes: {metrics['n_classes']}")
        print("=" * 50)
        print(f"\nArtifacts saved to: {args.out}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for the CLI.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        prog="modelgen",
        description="Generic CSV model generator with RandomForest baseline",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--train", type=Path, required=True, help="Path to training CSV file"
    )
    train_parser.add_argument(
        "--test", type=Path, help="Optional path to test CSV file"
    )
    train_parser.add_argument(
        "--task",
        choices=["reg", "cla"],
        required=True,
        help="Task type: 'reg' (regression) or 'cla' (classification)",
    )
    train_parser.add_argument(
        "--target", type=str, required=True, help="Target column name"
    )
    train_parser.add_argument(
        "--features",
        nargs="+",
        help="Feature column names (if not specified, use all except target)",
    )
    train_parser.add_argument(
        "--x-axis", type=str, help="Optional x-axis column for plots"
    )
    train_parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in RandomForest (default: 100)",
    )
    train_parser.add_argument(
        "--max-depth",
        type=int,
        help="Maximum tree depth (default: None, no limit)",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)",
    )
    train_parser.add_argument(
        "--ma-window",
        type=int,
        default=5,
        help="Moving average window size for plots (default: 5)",
    )
    train_parser.add_argument(
        "--out",
        type=Path,
        default=Path("output"),
        help="Output directory for artifacts (default: output)",
    )

    # Parse arguments
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Convert task abbreviation to full name
    if args.command == "train":
        args.task = "regression" if args.task == "reg" else "classification"
        return train_command(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
