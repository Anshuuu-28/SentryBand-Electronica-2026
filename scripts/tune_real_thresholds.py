"""
Module 1 (extension) -- Threshold tuning on REAL held-out data.

The first real-data bench run (68% accuracy) showed the heart classifier
over-triggering on real "fall" windows' resting PPG, dragging them into
"Combined Emergency" instead of "Possible Fall" -- almost certainly
because DEFAULT_HEART_THRESHOLD (config.py) was tuned against synthetic
PPG amplitude, not real PPG-DaLiA/MIMIC-AF amplitude. This script sweeps
fall_threshold x heart_threshold over real held-out windows and reports
full 4-class accuracy for every pair, picking the best.

Run AFTER scripts/train_and_bench_real.py (needs models_real/ to exist):
    python scripts/tune_real_thresholds.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from src.real_data_loader import build_real_dataset
from src.models import load_models
from src.fusion import fuse
from src.config import (
    STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT, STATE_COMBINED_EMERGENCY,
    RANDOM_SEED,
)

# Must match the paths you already set in train_and_bench_real.py
SISFALL_DIR = r"D:\Sentryband\raw_data\SisFall"
PPGDALIA_DIR = r"D:\Sentryband\raw_data\PPG-DaLiA\PPG_wrist_only"
MIMIC_AF_DIR = None

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_real"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

TRUE_STATE = {
    "normal": STATE_NORMAL, "fall": STATE_POSSIBLE_FALL,
    "heart": STATE_HEART_ALERT, "combined": STATE_COMBINED_EMERGENCY,
}
STATE_ORDER = [STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT, STATE_COMBINED_EMERGENCY]
THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def main():
    fall_clf, heart_clf = load_models(MODELS_DIR)

    # Same held-out test split train_and_bench_real.py used (seed+999),
    # so this tunes thresholds on the same real data, not new data.
    test_ds = build_real_dataset(SISFALL_DIR, PPGDALIA_DIR, n_per_class=50,
                                  seed=RANDOM_SEED + 999, mimic_af_dir=MIMIC_AF_DIR)

    fall_proba = fall_clf.predict_proba(test_ds.X_accel)[:, 1]
    heart_proba = heart_clf.predict_proba(test_ds.X_ppg)[:, 1]
    y_true = [TRUE_STATE[c] for c in test_ds.class_name]

    lines = [
        "SentryBand -- Real-Data Threshold Sweep",
        "=" * 45,
        f"{'fall_thr':>9s} {'heart_thr':>10s} {'accuracy':>10s}",
    ]

    results = []
    for fall_thr in THRESHOLDS:
        for heart_thr in THRESHOLDS:
            y_pred = [
                fuse(fp >= fall_thr, hp >= heart_thr)
                for fp, hp in zip(fall_proba, heart_proba)
            ]
            acc = accuracy_score(y_true, y_pred)
            results.append((fall_thr, heart_thr, acc))
            lines.append(f"{fall_thr:9.2f} {heart_thr:10.2f} {acc:10.3f}")

    best = max(results, key=lambda r: r[2])
    lines.append("")
    lines.append(f"Best: fall_threshold={best[0]:.2f}, heart_threshold={best[1]:.2f} "
                 f"-> accuracy={best[2]:.3f}")

    y_pred_best = [
        fuse(fp >= best[0], hp >= best[1])
        for fp, hp in zip(fall_proba, heart_proba)
    ]
    cm = confusion_matrix(y_true, y_pred_best, labels=STATE_ORDER)
    lines.append("")
    lines.append("Confusion matrix at best thresholds (rows = true, cols = predicted):")
    header = "                        " + "".join(f"{s[:12]:>14s}" for s in STATE_ORDER)
    lines.append(header)
    for i, s in enumerate(STATE_ORDER):
        row = "".join(f"{cm[i, j]:14d}" for j in range(len(STATE_ORDER)))
        lines.append(f"  true={s:20s}{row}")

    lines.append("")
    lines.append(f"Update src/config.py: DEFAULT_FALL_THRESHOLD = {best[0]:.2f}, "
                 f"DEFAULT_HEART_THRESHOLD = {best[1]:.2f}")

    report = "\n".join(lines)
    print(report)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "real_data_threshold_tuning_report.txt").write_text(report + "\n")
    print(f"\nReport saved to: {REPORTS_DIR / 'real_data_threshold_tuning_report.txt'}")


if __name__ == "__main__":
    main()