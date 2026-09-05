"""
Module 1 (extension) -- Multi-Fold Leave-Subject-Out Trial.

scripts/real_user_trials.py tests ONE held-out pair of real subjects.
With only 6 real PPG-DaLiA subjects available locally, a single draw can
be noisy. This script repeats the same leave-subjects-out test across
several different held-out pairs (rotating through the available real
subjects) and reports the mean and range -- a more defensible number
than any single fold.

This is the same idea as "k-fold Leave-One-Subject-Out (LOSO) cross-
validation" in the wearable-sensing research literature -- a standard,
named technique for exactly this situation, not something invented for
this project.

Run (same path setup as real_user_trials.py):
    python scripts/multifold_user_trials.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score

from src.real_data_loader import build_real_dataset
from src.models import train_classifiers
from src.fusion import fuse
from src.config import (
    RANDOM_SEED, DEFAULT_FALL_THRESHOLD, DEFAULT_HEART_THRESHOLD,
    STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT, STATE_COMBINED_EMERGENCY,
)
from real_user_trials import (
    SISFALL_DIR, PPGDALIA_DIR,
    discover_sisfall_subjects_with_fall_data, discover_ppgdalia_subjects,
)

N_FOLDS = 4
N_HELD_OUT_SISFALL_PER_FOLD = 2
N_HELD_OUT_PPGDALIA_PER_FOLD = 1  # only 6 real PPG-DaLiA subjects available locally

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

TRUE_STATE = {
    "normal": STATE_NORMAL, "fall": STATE_POSSIBLE_FALL,
    "heart": STATE_HEART_ALERT, "combined": STATE_COMBINED_EMERGENCY,
}


def run_fold(fold_idx, sisfall_holdout, ppgdalia_holdout):
    def train_sisfall_filter(stem):
        return not any(code in stem.upper() for code in sisfall_holdout)

    def train_ppgdalia_filter(stem):
        return stem.upper() not in ppgdalia_holdout

    def test_sisfall_filter(stem):
        return any(code in stem.upper() for code in sisfall_holdout)

    def test_ppgdalia_filter(stem):
        return stem.upper() in ppgdalia_holdout

    train_ds = build_real_dataset(
        SISFALL_DIR, PPGDALIA_DIR, n_per_class=100, seed=RANDOM_SEED + fold_idx,
        sisfall_subject_filter=train_sisfall_filter,
        ppgdalia_subject_filter=train_ppgdalia_filter,
    )
    fall_clf, heart_clf = train_classifiers(
        train_ds.X_accel, train_ds.y_fall, train_ds.X_ppg, train_ds.y_heart,
        seed=RANDOM_SEED + fold_idx,
    )
    test_ds = build_real_dataset(
        SISFALL_DIR, PPGDALIA_DIR, n_per_class=40, seed=RANDOM_SEED + 555 + fold_idx,
        sisfall_subject_filter=test_sisfall_filter,
        ppgdalia_subject_filter=test_ppgdalia_filter,
    )
    fall_proba = fall_clf.predict_proba(test_ds.X_accel)[:, 1]
    heart_proba = heart_clf.predict_proba(test_ds.X_ppg)[:, 1]
    y_true = [TRUE_STATE[c] for c in test_ds.class_name]
    y_pred = [
        fuse(fp >= DEFAULT_FALL_THRESHOLD, hp >= DEFAULT_HEART_THRESHOLD)
        for fp, hp in zip(fall_proba, heart_proba)
    ]
    return accuracy_score(y_true, y_pred)


def main():
    for label, path in [("SISFALL_DIR", SISFALL_DIR), ("PPGDALIA_DIR", PPGDALIA_DIR)]:
        if not Path(path).exists():
            print(f"ERROR: {label} = {path!r} does not exist. Edit the path "
                  "in real_user_trials.py (this script reuses it).")
            sys.exit(1)

    sisfall_eligible = discover_sisfall_subjects_with_fall_data(SISFALL_DIR)
    ppgdalia_subjects = discover_ppgdalia_subjects(PPGDALIA_DIR)
    print(f"{len(sisfall_eligible)} real SisFall subjects have both ADL+fall "
          f"data; {len(ppgdalia_subjects)} real PPG-DaLiA subjects available.")

    need_sisfall = N_FOLDS * N_HELD_OUT_SISFALL_PER_FOLD
    need_ppgdalia = N_FOLDS * N_HELD_OUT_PPGDALIA_PER_FOLD
    if len(sisfall_eligible) < need_sisfall or len(ppgdalia_subjects) < need_ppgdalia:
        print(f"Not enough distinct real subjects for {N_FOLDS} non-overlapping "
              f"folds (need {need_sisfall} SisFall + {need_ppgdalia} PPG-DaLiA "
              f"without reuse). Reduce N_FOLDS or the per-fold holdout counts.")
        sys.exit(1)

    accuracies = []
    lines = [
        "SentryBand -- Multi-Fold Leave-Subject-Out Trial",
        "=" * 50,
        f"{N_FOLDS} folds, each holding out {N_HELD_OUT_SISFALL_PER_FOLD} real "
        f"SisFall + {N_HELD_OUT_PPGDALIA_PER_FOLD} real PPG-DaLiA subject(s) "
        f"never used in that fold's training.",
        "",
    ]

    for fold in range(N_FOLDS):
        sf_holdout = sisfall_eligible[
            fold * N_HELD_OUT_SISFALL_PER_FOLD: (fold + 1) * N_HELD_OUT_SISFALL_PER_FOLD
        ]
        pg_holdout = ppgdalia_subjects[
            fold * N_HELD_OUT_PPGDALIA_PER_FOLD: (fold + 1) * N_HELD_OUT_PPGDALIA_PER_FOLD
        ]
        print(f"Fold {fold + 1}/{N_FOLDS}: holding out SisFall {sf_holdout}, "
              f"PPG-DaLiA {pg_holdout} ...")
        acc = run_fold(fold, sf_holdout, pg_holdout)
        accuracies.append(acc)
        line = f"Fold {fold + 1}: held out {sf_holdout} + {pg_holdout} -> accuracy {acc * 100:.2f}%"
        print("  " + line)
        lines.append(line)

    mean_acc = float(np.mean(accuracies))
    lines += [
        "",
        f"Mean leave-subject-out accuracy across {N_FOLDS} folds: {mean_acc * 100:.2f}%",
        f"Range: {min(accuracies) * 100:.2f}% - {max(accuracies) * 100:.2f}%",
        "",
        "For comparison: same-window (not same-subject-excluded) held-out "
        "accuracy was 76-77.5% (see reports/real_data_bench_report.txt).",
        "The gap between these two numbers IS the finding: it is a real, "
        "named challenge in wearable sensing research (Leave-One-Subject-Out "
        "generalization gap), not unique to this project, and an honest, "
        "specific target for future work (per-user calibration, more "
        "subject diversity in training, or domain-adaptation techniques).",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "multifold_user_trials_report.txt").write_text(report + "\n")
    print(f"\nReport saved to: {REPORTS_DIR / 'multifold_user_trials_report.txt'}")


if __name__ == "__main__":
    main()