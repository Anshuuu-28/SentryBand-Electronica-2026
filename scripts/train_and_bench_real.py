"""
Module 1 — Real Sensor Data: train + bench-test on REAL recordings from
SisFall (accelerometer/fall) and PPG-DaLiA (PPG/heart), instead of the
synthetic generators in src/sensors.py.

SETUP (one-time, free):
  1. Download SisFall (Kaggle mirror is easiest — search "SisFall") and
     unzip it somewhere.
  2. Download PPG-DaLiA data.zip from
     https://archive.ics.uci.edu/dataset/495/ppg+dalia and unzip it.
  3. Edit SISFALL_DIR and PPGDALIA_DIR below to point at the two folders.

Run from the project root:
    python scripts/train_and_bench_real.py

Outputs:
    models_real/fall_classifier.pkl, models_real/heart_classifier.pkl
    reports/real_data_bench_report.txt

This mirrors scripts/train.py + scripts/bench_test.py exactly, except the
train/test split comes from disjoint real recordings (different SisFall
trial files / different PPG-DaLiA subjects) instead of a different random
seed over a synthetic generator — a stronger, more honest held-out split.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

from src.real_data_loader import build_real_dataset
from src.models import train_classifiers, save_models, model_size_kb
from src.fusion import fuse
from src.config import (
    MODEL_FOOTPRINT_TARGET_KB, INFERENCE_LATENCY_TARGET_MS, RANDOM_SEED,
    STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT, STATE_COMBINED_EMERGENCY,
    DEFAULT_FALL_THRESHOLD, DEFAULT_HEART_THRESHOLD,
)

# <<< EDIT THESE PATHS after downloading the datasets >>>
SISFALL_DIR = r"D:\Sentryband\raw_data\SisFall"
PPGDALIA_DIR = r"D:\Sentryband\raw_data\PPG-DaLiA\PPG_wrist_only"
# Optional but recommended -- makes the Heart Alert class REAL recorded
# atrial fibrillation instead of a synthetic-timing hybrid. Download from
# https://zenodo.org/records/6967256 (mimic_perform_af_csv.zip, ~27 MB).
# Leave as None if you haven't downloaded it yet -- everything else still runs.
MIMIC_AF_DIR = None

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_real"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

TRUE_STATE = {
    "normal": STATE_NORMAL, "fall": STATE_POSSIBLE_FALL,
    "heart": STATE_HEART_ALERT, "combined": STATE_COMBINED_EMERGENCY,
}
STATE_ORDER = [STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT, STATE_COMBINED_EMERGENCY]


def main():
    for label, path in [("SISFALL_DIR", SISFALL_DIR), ("PPGDALIA_DIR", PPGDALIA_DIR)]:
        if not Path(path).exists():
            print(f"ERROR: {label} = {path!r} does not exist.")
            print("Edit the path at the top of this script after downloading "
                  "the dataset (see the module docstring / README).")
            sys.exit(1)

    print("Loading REAL SisFall + PPG-DaLiA data and building TRAIN split "
          "(seed=RANDOM_SEED)...")
    train_ds = build_real_dataset(SISFALL_DIR, PPGDALIA_DIR,
                                   n_per_class=100, seed=RANDOM_SEED,
                                   mimic_af_dir=MIMIC_AF_DIR)

    print("Building held-out TEST split (different seed => different "
          "sampled trials/subjects/windows)...")
    test_ds = build_real_dataset(SISFALL_DIR, PPGDALIA_DIR,
                                  n_per_class=50, seed=RANDOM_SEED + 999,
                                  mimic_af_dir=MIMIC_AF_DIR)

    print("Training fall + heart classifiers on real-data features...")
    fall_clf, heart_clf = train_classifiers(
        train_ds.X_accel, train_ds.y_fall, train_ds.X_ppg, train_ds.y_heart,
        seed=RANDOM_SEED,
    )
    MODELS_DIR.mkdir(exist_ok=True)
    save_models(fall_clf, heart_clf, MODELS_DIR)
    fall_kb, heart_kb = model_size_kb(fall_clf), model_size_kb(heart_clf)

    print("Running bench test on held-out real data...")
    fall_proba = fall_clf.predict_proba(test_ds.X_accel)[:, 1]
    heart_proba = heart_clf.predict_proba(test_ds.X_ppg)[:, 1]

    y_true = [TRUE_STATE[c] for c in test_ds.class_name]
    y_pred = [
        fuse(fp >= DEFAULT_FALL_THRESHOLD, hp >= DEFAULT_HEART_THRESHOLD)
        for fp, hp in zip(fall_proba, heart_proba)
    ]

    overall_acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=STATE_ORDER)

    lines = [
        "SentryBand -- Bench Test Report (REAL data: SisFall + PPG-DaLiA)",
        "=" * 66,
        f"Model footprint : fall {fall_kb:.2f} KB + heart {heart_kb:.2f} KB "
        f"= {fall_kb + heart_kb:.2f} KB (target < {MODEL_FOOTPRINT_TARGET_KB} KB)",
        f"Overall accuracy (held-out real data): {overall_acc * 100:.2f}%",
        "",
        "Confusion matrix (rows = true, cols = predicted):",
    ]
    header = "                        " + "".join(f"{s[:12]:>14s}" for s in STATE_ORDER)
    lines.append(header)
    for i, s in enumerate(STATE_ORDER):
        row = "".join(f"{cm[i, j]:14d}" for j in range(len(STATE_ORDER)))
        lines.append(f"  true={s:20s}{row}")
    heart_provenance = (
        "  - heart-alert PPG windows    : REAL MIMIC PERform AF recordings "
        "(genuine atrial fibrillation, Zenodo 6967256)"
        if MIMIC_AF_DIR else
        "  - heart-alert PPG windows    : real-beat-morphology extracted from\n"
        "    PPG-DaLiA, re-timed to tachycardia/bradycardia/arrhythmia rates\n"
        "    -- NOT real recorded arrhythmia data (set MIMIC_AF_DIR at the top\n"
        "    of this script to use real AF data instead, see module docstring)."
    )
    lines += [
        "",
        "Data provenance:",
        "  - accelerometer/fall windows : REAL SisFall recordings (waist-worn,",
        "    see CALIBRATION_SOURCES.md for the wrist-vs-waist caveat)",
        "  - normal/fall PPG windows    : REAL PPG-DaLiA wrist BVP recordings",
        heart_provenance,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "real_data_bench_report.txt").write_text(report + "\n")
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Report saved to: {REPORTS_DIR / 'real_data_bench_report.txt'}")


if __name__ == "__main__":
    main()
