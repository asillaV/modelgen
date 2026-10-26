"""Tests for CSV loading functionality."""

import tempfile
from pathlib import Path

import pytest

from modelgen.io.loaders import load_csv


def test_load_csv_basic():
    """Test basic CSV loading."""
    # Create temporary CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b,c\n1,2,3\n4,5,6\n")
        temp_path = Path(f.name)

    try:
        df, report = load_csv(temp_path)
        assert df.shape == (2, 3)
        assert list(df.columns) == ["a", "b", "c"]
        assert report.delimiter == ","
        assert report.header_row == 0
    finally:
        temp_path.unlink()


def test_load_csv_missing_file():
    """Test loading non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_csv(Path("nonexistent.csv"))


def test_single_numeric_column_is_not_split_on_decimal_point(tmp_path):
    """A one-column file of decimals must load as one float column."""
    from modelgen.io.loaders import load_csv

    path = tmp_path / "signal.csv"
    path.write_text("acc\n0.125\n-1.5\n2.75\n0.5\n")
    df, _ = load_csv(path)
    assert list(df.columns) == ["acc"]
    assert df["acc"].tolist() == [0.125, -1.5, 2.75, 0.5]
