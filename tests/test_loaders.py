"""Tests for CSV loading functionality."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from modelgen.io.loaders import detect_delimiter, detect_encoding, load_csv


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
