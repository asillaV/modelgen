"""Feature extraction from vibration signals for condition monitoring.

The functions in this module turn a raw 1-D signal (for example an
accelerometer on a bearing housing) into a table of features, one row per
analysis window. The table can then be used with the modelgen training
pipeline like any other CSV.

Three families of features are computed for every window:

* time domain: statistical descriptors of the waveform (RMS, peak, crest
  factor, kurtosis, ...). Kurtosis and crest factor react to the short
  impacts produced by a localised bearing defect.
* frequency domain: relative energy in equal-width frequency bands and the
  spectral centroid, which describe how the energy is distributed.
* envelope domain (optional): amplitude of the envelope spectrum at the
  characteristic defect frequencies of the bearing. A local defect excites a
  structural resonance at every impact, so the impact rate shows up in the
  envelope of the band-passed signal, not in the raw spectrum.
"""

from __future__ import annotations

import logging
from typing import Mapping, Optional

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats

logger = logging.getLogger(__name__)

# Defect frequencies as multiples of the shaft rotation frequency for the
# SKF 6205-2RS JEM bearing at the drive end of the Case Western Reserve
# University (CWRU) test rig, as published by the CWRU Bearing Data Center.
CWRU_6205_DE_ORDERS: dict[str, float] = {
    "bpfi": 5.4152,  # ball pass frequency, inner race
    "bpfo": 3.5848,  # ball pass frequency, outer race
    "bsf": 4.7135,  # ball spin frequency (rolling element)
    "ftf": 0.39828,  # fundamental train (cage) frequency
}


def sliding_windows(x: np.ndarray, window: int, step: int) -> np.ndarray:
    """Split a 1-D signal into windows of fixed length.

    Args:
        x: Input signal.
        window: Window length in samples.
        step: Distance in samples between the start of two windows.

    Returns:
        2-D array of shape (n_windows, window). The trailing samples that do
        not fill a whole window are discarded.

    Raises:
        ValueError: If the parameters are not positive or the signal is
            shorter than one window.
    """
    x = np.asarray(x, dtype=float).ravel()
    if window <= 0 or step <= 0:
        raise ValueError("window and step must be positive")
    if len(x) < window:
        raise ValueError(f"Signal of {len(x)} samples is shorter than window {window}")
    n = 1 + (len(x) - window) // step
    idx = np.arange(window)[None, :] + step * np.arange(n)[:, None]
    return x[idx]


def time_features(w: np.ndarray) -> dict[str, float]:
    """Time-domain descriptors of one window.

    Args:
        w: Window samples.

    Returns:
        Dictionary of features. Kurtosis uses the Pearson definition, so a
        Gaussian signal gives about 3.
    """
    w = np.asarray(w, dtype=float)
    abs_w = np.abs(w)
    rms = float(np.sqrt(np.mean(w**2)))
    peak = float(abs_w.max())
    mean_abs = float(abs_w.mean())
    sqrt_mean = float(np.mean(np.sqrt(abs_w)) ** 2)
    eps = np.finfo(float).eps
    return {
        "mean": float(w.mean()),
        "std": float(w.std()),
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(w.max() - w.min()),
        "crest_factor": peak / (rms + eps),
        "kurtosis": float(stats.kurtosis(w, fisher=False)),
        "skewness": float(stats.skew(w)),
        "shape_factor": rms / (mean_abs + eps),
        "impulse_factor": peak / (mean_abs + eps),
        "clearance_factor": peak / (sqrt_mean + eps),
    }


def spectral_features(w: np.ndarray, fs: float, n_bands: int = 8) -> dict[str, float]:
    """Frequency-domain descriptors of one window.

    Args:
        w: Window samples.
        fs: Sampling frequency in Hz.
        n_bands: Number of equal-width bands between 0 and fs/2.

    Returns:
        Spectral centroid in Hz and the fraction of total power in each band
        (``band_00`` is the lowest band). The band fractions sum to 1.
    """
    w = np.asarray(w, dtype=float)
    w = (w - w.mean()) * np.hanning(len(w))
    power = np.abs(np.fft.rfft(w)) ** 2
    freqs = np.fft.rfftfreq(len(w), d=1.0 / fs)
    total = power.sum() + np.finfo(float).eps

    feats: dict[str, float] = {
        "spectral_centroid_hz": float((freqs * power).sum() / total),
    }
    edges = np.linspace(0.0, fs / 2.0, n_bands + 1)
    band_index = np.clip(
        np.searchsorted(edges, freqs, side="right") - 1, 0, n_bands - 1
    )
    band_power = np.bincount(band_index, weights=power, minlength=n_bands)
    for i, p in enumerate(band_power):
        feats[f"band_{i:02d}"] = float(p / total)
    return feats


def envelope_spectrum(
    w: np.ndarray, fs: float, band: Optional[tuple[float, float]] = None
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope spectrum of one window.

    The window is band-passed around the resonance excited by the impacts,
    demodulated with the Hilbert transform, and the spectrum of the envelope
    is returned.

    Args:
        w: Window samples.
        fs: Sampling frequency in Hz.
        band: Pass band (low, high) in Hz. Defaults to 5%-40% of fs.

    Returns:
        Tuple (frequencies in Hz, amplitude spectrum of the envelope).
    """
    lo, hi = band if band is not None else (0.05 * fs, 0.40 * fs)
    if not 0 < lo < hi < fs / 2:
        raise ValueError(f"Invalid envelope band {band} for fs={fs}")
    sos = sps.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    filtered = sps.sosfiltfilt(sos, np.asarray(w, dtype=float))
    env = np.abs(sps.hilbert(filtered))
    env = (env - env.mean()) * np.hanning(len(env))
    amp = np.abs(np.fft.rfft(env)) * 2.0 / len(env)
    freqs = np.fft.rfftfreq(len(env), d=1.0 / fs)
    return freqs, amp


def envelope_features(
    w: np.ndarray,
    fs: float,
    shaft_hz: float,
    defect_orders: Mapping[str, float],
    band: Optional[tuple[float, float]] = None,
    harmonics: int = 2,
) -> dict[str, float]:
    """Envelope amplitude at the bearing defect frequencies.

    For each defect frequency f = order * shaft_hz the peak of the envelope
    spectrum near f and its harmonics is summed and divided by the median
    envelope amplitude, so the value is a dimensionless "how much does this
    line stand out" ratio that does not depend on the overall vibration
    level.

    Args:
        w: Window samples.
        fs: Sampling frequency in Hz.
        shaft_hz: Shaft rotation frequency in Hz.
        defect_orders: Mapping name -> multiple of the shaft frequency, for
            example ``CWRU_6205_DE_ORDERS``.
        band: Pass band for the envelope, see ``envelope_spectrum``.
        harmonics: Number of harmonics summed for each defect frequency.

    Returns:
        Dictionary with one ``env_<name>`` feature per defect frequency.
    """
    freqs, amp = envelope_spectrum(w, fs, band)
    df = freqs[1] - freqs[0]
    f_max = max(defect_orders.values()) * shaft_hz * (harmonics + 1)
    reference = (
        float(np.median(amp[(freqs > 0) & (freqs <= f_max)])) + np.finfo(float).eps
    )

    feats: dict[str, float] = {}
    for name, order in defect_orders.items():
        total = 0.0
        for h in range(1, harmonics + 1):
            f0 = h * order * shaft_hz
            tol = max(2.0 * df, 0.02 * f0)
            sel = (freqs >= f0 - tol) & (freqs <= f0 + tol)
            if sel.any():
                total += float(amp[sel].max())
        feats[f"env_{name}"] = total / reference
    return feats


def extract_features(
    x: np.ndarray,
    fs: float,
    window: int = 4096,
    step: Optional[int] = None,
    n_bands: int = 8,
    shaft_hz: Optional[float] = None,
    defect_orders: Optional[Mapping[str, float]] = None,
    envelope_band: Optional[tuple[float, float]] = None,
) -> pd.DataFrame:
    """Compute the feature table of a signal, one row per window.

    Args:
        x: Input signal.
        fs: Sampling frequency in Hz.
        window: Window length in samples.
        step: Step between windows in samples. Defaults to ``window``
            (no overlap).
        n_bands: Number of spectral bands.
        shaft_hz: Shaft frequency in Hz. Envelope features are computed only
            when both ``shaft_hz`` and ``defect_orders`` are given.
        defect_orders: Defect frequencies as multiples of the shaft frequency.
        envelope_band: Pass band for envelope analysis.

    Returns:
        DataFrame with a ``window_start_s`` column followed by the features.
    """
    step = step or window
    windows = sliding_windows(x, window, step)
    use_envelope = shaft_hz is not None and defect_orders is not None

    rows = []
    for i, w in enumerate(windows):
        row: dict[str, float] = {"window_start_s": i * step / fs}
        row.update(time_features(w))
        row.update(spectral_features(w, fs, n_bands))
        if use_envelope:
            row.update(envelope_features(w, fs, shaft_hz, defect_orders, envelope_band))
        rows.append(row)

    logger.info(
        f"Extracted {len(rows)} windows x {len(rows[0]) - 1} features "
        f"(window={window}, step={step}, fs={fs})"
    )
    return pd.DataFrame(rows)
