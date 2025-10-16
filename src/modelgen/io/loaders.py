"""CSV loading utilities with automatic detection of encoding, delimiter, and header."""

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import chardet
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LoadReport:
    """Report containing metadata about the loaded CSV file.

    Attributes:
        delimiter: Detected delimiter character.
        encoding: Detected encoding.
        header_row: Row index of header (0) or None if no header.
        shape: Tuple of (rows, columns) in the DataFrame.
        na_pct: Percentage of missing values in the DataFrame.
    """

    delimiter: str
    encoding: str
    header_row: Optional[int]
    shape: tuple[int, int]
    na_pct: float


def detect_encoding(file_path: Path, sample_size: int = 10000) -> str:
    """Detect file encoding using chardet.

    Args:
        file_path: Path to the CSV file.
        sample_size: Number of bytes to read for detection.

    Returns:
        Detected encoding string (e.g., 'utf-8', 'utf-16').

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        raw_data = f.read(sample_size)

    result = chardet.detect(raw_data)
    detected_encoding = result["encoding"]
    confidence = result["confidence"]

    logger.info(
        f"Detected encoding: {detected_encoding} (confidence: {confidence:.2f})"
    )

    # Fallback to utf-8 if confidence is low
    if confidence < 0.7:
        logger.warning(
            f"Low confidence ({confidence:.2f}), falling back to utf-8"
        )
        return "utf-8"

    return detected_encoding or "utf-8"


def detect_delimiter(file_path: Path, encoding: str) -> str:
    """Detect CSV delimiter using csv.Sniffer.

    Args:
        file_path: Path to the CSV file.
        encoding: File encoding to use when reading.

    Returns:
        Detected delimiter character (e.g., ',', ';', '\t').

    Raises:
        ValueError: If delimiter detection fails.
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            sample = f.read(8192)
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            logger.info(f"Detected delimiter: '{delimiter}'")
            return delimiter
    except Exception as e:
        logger.warning(f"Delimiter detection failed: {e}. Using default ','")
        return ","


def detect_header(file_path: Path, encoding: str, delimiter: str) -> Optional[int]:
    """Detect if the CSV has a header row.

    Args:
        file_path: Path to the CSV file.
        encoding: File encoding to use when reading.
        delimiter: Delimiter character.

    Returns:
        0 if header is detected, None otherwise.
    """
    try:
        with open(file_path, "r", encoding=encoding) as f:
            sample = f.read(8192)
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(sample)
            logger.info(f"Header detected: {has_header}")
            return 0 if has_header else None
    except Exception as e:
        logger.warning(f"Header detection failed: {e}. Assuming header exists.")
        return 0


def load_csv(
    path_or_buffer: Union[str, Path],
    *,
    sample_rows: Optional[int] = None,
    encoding: Optional[str] = None,
    delimiter: Optional[str] = None,
    header: Optional[int] = 0,
) -> tuple[pd.DataFrame, LoadReport]:
    """Load CSV file with automatic detection of encoding, delimiter, and header.

    If encoding, delimiter, or header are not provided, they will be auto-detected.

    Args:
        path_or_buffer: Path to the CSV file or file-like object.
        sample_rows: Optional number of rows to load (for large files).
        encoding: Optional encoding override. If None, auto-detect.
        delimiter: Optional delimiter override. If None, auto-detect.
        header: Optional header row index. If None, auto-detect.

    Returns:
        Tuple of (DataFrame, LoadReport) containing the data and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the CSV cannot be parsed.

    Example:
        >>> df, report = load_csv("data.csv", sample_rows=5000)
        >>> print(f"Loaded {report.shape[0]} rows with {report.na_pct:.1f}% missing")
    """
    file_path = Path(path_or_buffer) if isinstance(path_or_buffer, str) else path_or_buffer

    if isinstance(file_path, Path) and not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Auto-detect parameters if not provided
    if encoding is None and isinstance(file_path, Path):
        encoding = detect_encoding(file_path)
    elif encoding is None:
        encoding = "utf-8"

    if delimiter is None and isinstance(file_path, Path):
        delimiter = detect_delimiter(file_path, encoding)
    elif delimiter is None:
        delimiter = ","

    if header == 0 and isinstance(file_path, Path):
        detected_header = detect_header(file_path, encoding, delimiter)
        if detected_header is None:
            header = None

    # Load the CSV
    try:
        logger.info(f"Loading CSV: {file_path}")
        df = pd.read_csv(
            file_path,
            sep=delimiter,
            encoding=encoding,
            header=header,
            nrows=sample_rows,
            low_memory=False,
        )
        logger.info(f"Loaded DataFrame with shape {df.shape}")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise ValueError(f"Failed to parse CSV file: {e}")

    # Calculate missing value percentage
    total_values = df.shape[0] * df.shape[1]
    missing_values = df.isna().sum().sum()
    na_pct = (missing_values / total_values * 100) if total_values > 0 else 0.0

    # Create load report
    report = LoadReport(
        delimiter=delimiter,
        encoding=encoding,
        header_row=header,
        shape=df.shape,
        na_pct=na_pct,
    )

    logger.info(f"Load report: {report}")

    return df, report
