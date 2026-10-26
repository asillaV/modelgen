"""Bearing fault classification on CWRU features, evaluated honestly.

Classes: normal, inner_race, ball, outer_race (drive-end bearing, 48 kHz).

Two evaluation protocols are compared:

* random split: windows are shuffled and split 80/20. Windows of the same
  recording (and, with 50% overlap, partly the same samples) end up in both
  train and test, so the score mostly measures memorisation. Many public
  CWRU results are obtained this way.
* leave-one-load-out: the model is trained on three motor loads and tested on
  the fourth, which it has never seen, then the four folds are averaged. No
  recording appears on both sides. This is closer to deploying a model on a
  machine running at a new operating point.

Each protocol is run with time + spectral features only and with the
envelope features at the bearing defect frequencies added.

Usage:
    python examples/predictive_maintenance/run_experiment.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split

from modelgen.ml.evaluate import evaluate_classification
from modelgen.ml.train import create_model

HERE = Path(__file__).parent
META = ["recording", "fault", "size_mils", "load_hp", "rpm", "window_start_s"]
CLASSES = ["normal", "inner_race", "ball", "outer_race"]


def make_model(n_estimators: int, seed: int):
    model = create_model("classification", n_estimators=n_estimators, random_state=seed)
    return model.set_params(class_weight="balanced")


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    features = [c for c in df.columns if c not in META]
    base = [c for c in features if not c.startswith("env_")]
    return {"time + spectral": base, "time + spectral + envelope": features}


def random_split(df, cols, n_estimators, seed):
    train, test = train_test_split(
        df, test_size=0.2, stratify=df["fault"], random_state=seed
    )
    model = make_model(n_estimators, seed).fit(train[cols], train["fault"])
    pred = model.predict(test[cols])
    return {
        "f1_macro": f1_score(test["fault"], pred, average="macro"),
        "accuracy": float((pred == test["fault"]).mean()),
    }


def leave_one_load_out(df, cols, n_estimators, seed):
    folds, y_true, y_pred = [], [], []
    for load in sorted(df["load_hp"].unique()):
        train, test = df[df["load_hp"] != load], df[df["load_hp"] == load]
        assert not set(train["recording"]) & set(test["recording"])
        model = make_model(n_estimators, seed).fit(train[cols], train["fault"])
        pred = model.predict(test[cols])
        folds.append(
            {
                "test_load_hp": int(load),
                "f1_macro": f1_score(test["fault"], pred, average="macro"),
                "accuracy": float((pred == test["fault"]).mean()),
            }
        )
        y_true.extend(test["fault"])
        y_pred.extend(pred)
    return (
        {
            "f1_macro": float(np.mean([f["f1_macro"] for f in folds])),
            "f1_macro_min": float(np.min([f["f1_macro"] for f in folds])),
            "accuracy": float(np.mean([f["accuracy"] for f in folds])),
            "folds": folds,
            "pooled": evaluate_classification(pd.Series(y_true), pd.Series(y_pred)),
        },
        y_true,
        y_pred,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="CWRU bearing fault experiment")
    parser.add_argument("--features", type=Path, default=HERE / "cwru_features.csv")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} windows, {df['recording'].nunique()} recordings")
    print(
        df.groupby("fault")["recording"].nunique().rename("recordings").to_string(),
        "\n",
    )

    results, best = {}, None
    for name, cols in feature_sets(df).items():
        rnd = random_split(df, cols, args.n_estimators, args.seed)
        lolo, y_true, y_pred = leave_one_load_out(
            df, cols, args.n_estimators, args.seed
        )
        results[name] = {
            "n_features": len(cols),
            "random_split": rnd,
            "leave_one_load_out": lolo,
        }
        best = (name, cols, y_true, y_pred)

    lines = [
        "| Features | Random split F1 | Leave-one-load-out F1 (mean / worst fold) |",
        "|---|---|---|",
    ]
    for name, r in results.items():
        lolo = r["leave_one_load_out"]
        lines.append(
            f"| {name} ({r['n_features']}) | {r['random_split']['f1_macro']:.3f} | "
            f"{lolo['f1_macro']:.3f} / {lolo['f1_macro_min']:.3f} |"
        )
    table = "\n".join(lines)
    print(table)

    # Figures for the full feature set
    name, cols, y_true, y_pred = best
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=CLASSES, ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Leave-one-load-out, all folds pooled")
    fig.tight_layout()
    fig.savefig(args.out / "confusion_leave_one_load_out.png", dpi=130)

    model = make_model(args.n_estimators, args.seed).fit(df[cols], df["fault"])
    imp = pd.Series(model.feature_importances_, index=cols).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    imp.plot.barh(ax=ax, color="#3b6ea5")
    ax.set_xlabel("Impurity-based importance")
    ax.set_title("Top features (model trained on all loads)")
    fig.tight_layout()
    fig.savefig(args.out / "feature_importance.png", dpi=130)

    (args.out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    (args.out / "results_table.md").write_text(table + "\n")
    print(f"\nResults saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
