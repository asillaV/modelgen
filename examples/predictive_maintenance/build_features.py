"""Build the feature table for the CWRU bearing example.

Reads the drive-end accelerometer of the CWRU Bearing Data Center recordings
sampled at 48 kHz (normal baseline plus inner race, ball and outer race
faults at every motor load) and writes one row per analysis window.

The raw .mat files are not stored in this repository. Download them from the
CWRU Bearing Data Center (https://engineering.case.edu/bearingdatacenter) or
from a mirror that keeps the original folder layout, for example:

    git clone --depth 1 https://github.com/s-whynot/CWRU-dataset data/cwru

then run:

    python examples/predictive_maintenance/build_features.py \
        --data-dir data/cwru \
        --out examples/predictive_maintenance/cwru_features.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from scipy.io import loadmat

from modelgen.signal.features import CWRU_6205_DE_ORDERS, extract_features

FS = 48_000.0
WINDOW = 8192  # ~0.17 s: about 5 shaft revolutions, 5.9 Hz spectral resolution
STEP = 4096  # 50% overlap


def recordings(data_dir: Path) -> list[dict]:
    """List the recordings used in the example with their metadata."""
    items = []
    for path in sorted((data_dir / "Normal").glob("*.mat")):
        load = int(re.search(r"_(\d)\.mat$", path.name).group(1))
        items.append({"path": path, "fault": "normal", "size_mils": 0, "load_hp": load})

    fault_root = data_dir / "48k_Drive_End_Bearing_Fault_Data"
    names = {"IR": "inner_race", "B": "ball", "OR": "outer_race"}
    for path in sorted(fault_root.rglob("*.mat")):
        parts = path.relative_to(fault_root).parts
        load = int(re.search(r"_(\d)\.mat$", path.name).group(1))
        items.append(
            {
                "path": path,
                "fault": names[parts[0]],
                "size_mils": int(parts[1]),
                "load_hp": load,
            }
        )
    return items


def read_recording(path: Path) -> tuple[pd.Series, float]:
    """Return the drive-end signal and the shaft speed in rpm."""
    mat = loadmat(path)
    de_key = next(k for k in mat if k.endswith("_DE_time"))
    rpm_keys = [k for k in mat if k.endswith("RPM")]
    rpm = float(mat[rpm_keys[0]].ravel()[0]) if rpm_keys else float("nan")
    return mat[de_key].ravel(), rpm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tables = []
    for rec in recordings(args.data_dir):
        x, rpm = read_recording(rec["path"])
        # A few files have no RPM entry: fall back to the nominal speed for
        # that motor load reported by CWRU.
        if not rpm == rpm:  # NaN check
            rpm = {0: 1797, 1: 1772, 2: 1750, 3: 1730}[rec["load_hp"]]
        feats = extract_features(
            x,
            fs=FS,
            window=WINDOW,
            step=STEP,
            shaft_hz=rpm / 60.0,
            defect_orders=CWRU_6205_DE_ORDERS,
        )
        feats.insert(0, "recording", rec["path"].stem)
        feats.insert(1, "fault", rec["fault"])
        feats.insert(2, "size_mils", rec["size_mils"])
        feats.insert(3, "load_hp", rec["load_hp"])
        feats.insert(4, "rpm", rpm)
        tables.append(feats)
        print(
            f"{rec['path'].name:>16}  {rec['fault']:<10} load {rec['load_hp']}  "
            f"rpm {rpm:.0f}  windows {len(feats)}"
        )

    out = pd.concat(tables, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, float_format="%.6g")
    print(f"\n{len(out)} windows from {len(tables)} recordings -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
