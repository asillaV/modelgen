"""Tests for vibration feature extraction."""

import numpy as np
import pytest

from modelgen.signal.features import (
    envelope_features,
    extract_features,
    sliding_windows,
    spectral_features,
    time_features,
)

FS = 12_000.0


def test_sliding_windows_shape_and_content():
    x = np.arange(10)
    w = sliding_windows(x, window=4, step=3)
    assert w.shape == (3, 4)
    assert w[1].tolist() == [3, 4, 5, 6]


def test_sliding_windows_rejects_short_signal():
    with pytest.raises(ValueError):
        sliding_windows(np.zeros(3), window=4, step=1)


def test_time_features_of_sine():
    t = np.arange(int(FS)) / FS
    x = 2.0 * np.sin(2 * np.pi * 50 * t)
    f = time_features(x)
    assert f["rms"] == pytest.approx(2.0 / np.sqrt(2), rel=1e-3)
    assert f["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-3)
    # A pure sine has Pearson kurtosis 1.5
    assert f["kurtosis"] == pytest.approx(1.5, rel=1e-2)


def test_impacts_raise_kurtosis():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 8192)
    impacts = noise.copy()
    impacts[::800] += 15.0
    assert time_features(impacts)["kurtosis"] > time_features(noise)["kurtosis"] + 3


def test_spectral_bands_sum_to_one_and_find_tone():
    t = np.arange(4096) / FS
    x = np.sin(2 * np.pi * 4500 * t)
    f = spectral_features(x, FS, n_bands=6)
    bands = [f[f"band_{i:02d}"] for i in range(6)]
    assert sum(bands) == pytest.approx(1.0, rel=1e-9)
    # 4500 Hz falls in band 4 (4000-5000 Hz) of 6 equal bands over 0-6000 Hz
    assert int(np.argmax(bands)) == 4


def _impact_train(rate_hz, n=16384, resonance=3000.0, seed=1):
    """Decaying resonance bursts repeated at rate_hz, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    x = rng.normal(0, 0.2, n)
    burst_t = np.arange(int(0.004 * FS)) / FS
    burst = np.exp(-burst_t * 1500) * np.sin(2 * np.pi * resonance * burst_t)
    for start in np.arange(0, t[-1], 1.0 / rate_hz):
        i = int(start * FS)
        seg = burst[: n - i]
        x[i : i + len(seg)] += seg
    return x


def test_envelope_points_to_the_right_defect():
    shaft_hz = 25.0
    orders = {"outer": 3.5, "inner": 5.4}
    x = _impact_train(rate_hz=orders["outer"] * shaft_hz)
    f = envelope_features(x, FS, shaft_hz, orders, band=(2000, 4000))
    assert f["env_outer"] > 3 * f["env_inner"]


def test_extract_features_table():
    x = _impact_train(rate_hz=90.0, n=20000)
    df = extract_features(
        x,
        FS,
        window=4096,
        step=2048,
        shaft_hz=25.0,
        defect_orders={"outer": 3.6},
        envelope_band=(2000, 4000),
    )
    assert len(df) == 1 + (20000 - 4096) // 2048
    assert {"rms", "kurtosis", "band_00", "env_outer", "window_start_s"} <= set(
        df.columns
    )
    assert not df.isna().any().any()
