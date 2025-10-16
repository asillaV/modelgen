"""Generate synthetic CSV datasets for examples."""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_regression_data(n_samples: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic regression dataset.

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with features (x1, x2, x3) and target.
    """
    np.random.seed(random_state)

    x1 = np.random.normal(10, 5, n_samples)
    x2 = np.random.normal(20, 10, n_samples)
    x3 = np.random.uniform(0, 50, n_samples)

    # Target is a noisy linear combination
    target = 2.5 * x1 + 1.8 * x2 - 0.5 * x3 + np.random.normal(0, 10, n_samples)

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "target": target})

    return df


def generate_classification_data(
    n_samples: int = 200, random_state: int = 42
) -> pd.DataFrame:
    """Generate synthetic classification dataset.

    Args:
        n_samples: Number of samples to generate.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with features (x1, x2) and label (A or B).
    """
    np.random.seed(random_state)

    # Generate two clusters
    n_class_a = n_samples // 2
    n_class_b = n_samples - n_class_a

    # Class A: centered around (10, 10)
    x1_a = np.random.normal(10, 3, n_class_a)
    x2_a = np.random.normal(10, 3, n_class_a)

    # Class B: centered around (25, 25)
    x1_b = np.random.normal(25, 3, n_class_b)
    x2_b = np.random.normal(25, 3, n_class_b)

    x1 = np.concatenate([x1_a, x1_b])
    x2 = np.concatenate([x2_a, x2_b])
    label = ["A"] * n_class_a + ["B"] * n_class_b

    df = pd.DataFrame({"x1": x1, "x2": x2, "label": label})

    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


def main():
    """Generate and save example datasets."""
    # Create examples directory
    examples_dir = Path(__file__).parent.parent / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Generate regression dataset
    print("Generating regression_example.csv...")
    df_reg = generate_regression_data()
    reg_path = examples_dir / "regression_example.csv"
    df_reg.to_csv(reg_path, index=False)
    print(f"Saved to {reg_path}")
    print(f"  Shape: {df_reg.shape}")
    print(f"  Columns: {list(df_reg.columns)}")

    # Generate classification dataset
    print("\nGenerating classification_example.csv...")
    df_cla = generate_classification_data()
    cla_path = examples_dir / "classification_example.csv"
    df_cla.to_csv(cla_path, index=False)
    print(f"Saved to {cla_path}")
    print(f"  Shape: {df_cla.shape}")
    print(f"  Columns: {list(df_cla.columns)}")
    print(f"  Classes: {df_cla['label'].unique()}")

    print("\nDone!")


if __name__ == "__main__":
    main()
