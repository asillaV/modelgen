# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.2.0] - 2026-10-26

### Added
- `modelgen.signal.features`: windowed vibration features (time domain, spectral bands, envelope amplitude at bearing defect frequencies)
- `modelgen features` CLI command: signal CSV to feature CSV
- Predictive maintenance example on the CWRU bearing dataset with random split vs leave-one-load-out evaluation
- Tests for feature extraction and for the example
- GitHub Actions CI running the test suite on Python 3.11 and 3.12

### Fixed
- CSV delimiter detection split single-column numeric files on the decimal point

## [0.1.0] - 2025-01-XX

### Added
- Initial release of ModelGen
- CSV loading with automatic encoding, delimiter, and header detection
- Column type inference (numeric, categorical, datetime)
- Preprocessing pipelines with ColumnTransformer
- RandomForest training for regression and classification tasks
- Evaluation metrics (R²/MSE/MAE for regression, Accuracy/F1 for classification)
- Artifact management (model.pkl, predictions.csv, report.json, report.png)
- Visualization utilities (overlay plots, moving average, feature importance)
- Command-line interface with argparse
- GUI application with customtkinter (tkinter fallback)
- Example synthetic datasets for regression and classification
- Unit tests for core functionality
- Pre-commit hooks for code quality (black, ruff)
- Comprehensive documentation (README, docstrings)

### Features
- Python 3.11+ support
- Type hints throughout codebase
- Structured logging
- Reproducible results (fixed random_state=42)
- MIT License
