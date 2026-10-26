"""Smoke test: the CWRU example runs end to end on the committed features."""

import json
import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).parents[1] / "examples" / "predictive_maintenance"


def test_experiment_runs_and_separates_protocols(tmp_path):
    subprocess.run(
        [
            sys.executable,
            str(EXAMPLE / "run_experiment.py"),
            "--n-estimators",
            "30",
            "--out",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
    )
    results = json.loads((tmp_path / "results.json").read_text())
    for r in results.values():
        lolo = r["leave_one_load_out"]
        assert len(lolo["folds"]) == 4
        # Normal vs faulty must stay separable on unseen loads
        cm = lolo["pooled"]["confusion_matrix"]
        assert cm["normal"]["normal"] == sum(cm["normal"].values())
    assert (tmp_path / "confusion_leave_one_load_out.png").exists()
