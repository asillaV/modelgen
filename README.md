# ModelGen

A generic CSV model generator with RandomForest baseline for regression and classification tasks.

## Overview

ModelGen is a self-contained tool for rapid prototyping of machine learning models from CSV data. It provides automatic CSV ingestion with encoding/delimiter detection, preprocessing pipelines, RandomForest training, evaluation metrics, and visualization. Designed for demonstration and portfolio purposes, not production deployment.

## GUI Preview

![ModelGen GUI Preview](assets/GUI_example.png)

## Features

- **Robust CSV Ingestion**: Auto-detect encoding (chardet), delimiter (csv.Sniffer), and header presence
- **Dual Task Support**: Regression (R², MSE, MAE) and Classification (Accuracy, F1-Macro, confusion matrix)
- **Preprocessing Pipeline**: Automatic handling of numeric (median imputation) and categorical (most_frequent + OneHotEncoding) features
- **Model Training**: RandomForest with configurable n_estimators (50-500) and max_depth (None/2-20)
- **Comprehensive Artifacts**: Saves model.pkl, predictions.csv, report.json, and report.png
- **Visualization**: Overlay plots, moving averages, and feature importance charts
- **Dual Interface**: Command-line (argparse) and GUI (customtkinter with tkinter fallback)

## Installation

Requires Python 3.11 or higher.

```bash
# Clone the repository
cd modelgen

# Install the package in editable mode
pip install -e .

# Optional: Install GUI dependencies
pip install -e ".[gui]"

# Optional: Install development dependencies
pip install -e ".[dev]"
```

Alternatively, install from requirements.txt:

```bash
pip install -r requirements.txt
```

## Quickstart

### CLI Examples

**Regression:**
```bash
modelgen train \
  --train examples/regression_example.csv \
  --task reg \
  --target target \
  --features x1 x2 x3 \
  --n-estimators 100 \
  --out output/regression
```

**Classification:**
```bash
modelgen train \
  --train examples/classification_example.csv \
  --task cla \
  --target label \
  --features x1 x2 \
  --n-estimators 100 \
  --out output/classification
```

### GUI

Launch the graphical interface:

```bash
python -m modelgen.ui_app
```

Steps:
1. Select training CSV and click "Load Columns"
2. Choose task type (Regression/Classification)
3. Select target and features from dropdowns/listbox
4. Adjust n_estimators and max_depth sliders
5. Click "Train Model"

Results appear in the log panel and artifacts are saved to `output/`.

## Project Structure

```
modelgen/
├─ src/modelgen/
│  ├─ __init__.py
│  ├─ cli.py                # CLI entry point with argparse
│  ├─ ui_app.py             # GUI application (customtkinter/tkinter)
│  ├─ configs.py            # Dataclasses for configuration
│  ├─ logging_setup.py      # Logging initialization
│  ├─ io/
│  │  ├─ loaders.py         # CSV loading with auto-detect (encoding, delimiter, header)
│  │  └─ schema.py          # Column type inference (numeric/categorical/datetime)
│  ├─ ml/
│  │  ├─ pipelines.py       # ColumnTransformer with numeric/categorical preprocessing
│  │  ├─ train.py           # RandomForest training (regression/classification)
│  │  ├─ evaluate.py        # Metrics calculation (R²/MSE/MAE for reg, Acc/F1 for cla)
│  │  └─ artifacts.py       # Save/load model, predictions, report
│  └─ viz/
│     └─ plots.py           # Visualization (overlay, moving average, feature importance)
├─ examples/
│  ├─ regression_example.csv
│  └─ classification_example.csv
├─ tests/
│  ├─ test_loaders.py
│  └─ test_train_eval.py
├─ scripts/
│  └─ generate_synthetic_data.py
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

## Design Notes

**CSV Ingestion:**
Uses `chardet` for encoding detection (fallback to utf-8), `csv.Sniffer` for delimiter/header inference. User can override via optional parameters. Returns `LoadReport` dataclass with metadata.

**Preprocessing Pipeline:**
`ColumnTransformer` separates numeric and categorical features. Numeric: median imputation. Categorical: most_frequent imputation + OneHotEncoding (max_categories=32, handle_unknown="ignore").

**Model Training:**
Single-pass RandomForest training (no warm_start loops) with fixed random_state=42 for reproducibility. Supports n_estimators (50-500) and max_depth (None or 2-20).

**Evaluation:**
Regression: R², MSE, MAE, RMSE. Classification: Accuracy, F1-Macro, confusion matrix (as dict). Metrics logged and saved to report.json.

**Artifacts:**
All outputs saved to specified directory: model.pkl (joblib), predictions.csv, report.json (parameters + metrics + timings), report.png (composite figure with 3 subplots for regression).

**Logging:**
Structured logging to console and optional file (INFO level default). All library code uses logging, not print statements.

## Limitations

- Baseline model only (RandomForest). No hyperparameter tuning, ensemble stacking, or neural networks.
- Single train/test split. No cross-validation or stratified sampling.
- No MLOps features (model versioning, deployment, monitoring).
- Not suitable for confidential or sensitive data (local processing only).
- Performance optimized for 10-50k rows. Larger datasets may require sampling.

## License

MIT License. See LICENSE file for details.

## Contributing

This is a demonstration project. For issues or feature requests, open an issue on the repository.

## Example Output

After running `modelgen train`, you will see:

```
Training Results:
==================================================
R² Score: 0.9234
MSE: 12.45
MAE: 2.31
RMSE: 3.53
==================================================

Artifacts saved to: output/
```

Check `output/` for:
- `model.pkl`: Trained pipeline
- `predictions.csv`: Test set predictions
- `report.json`: Full training report with parameters and metrics
- `report.png`: Visualization (regression only)
