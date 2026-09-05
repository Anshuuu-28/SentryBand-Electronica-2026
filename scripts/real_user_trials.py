"""
Module 1 (extension) -- Real User Trials.

Replaces the synthetic "noise-perturbed users" stand-in
(scripts/user_trials.py) with a genuine test: train on most real SisFall
+ PPG-DaLiA subjects, then test ONLY on a real subject the model never
saw during training -- an honest measure of generalization to a new
real person, not synthetic noise.

Run AFTER you have real datasets set up (same paths as
scripts/train_and_bench_real.py):
    python scripts/real_user_trials.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score

from src.real_data_loader import build_real_dataset, load_sisfall_accel_windows, load_ppgdalia_normal_windows
from src.models import train_classifiers
from src.fusion import fuse
from src.config import (
    RANDOM_SEED, DEFAULT_FALL_THRESHOLD, DEFAULT_HEART_THRESHOLD,
    STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT, STATE_COMBINED_EMERGENCY,
)

# <<< Must match the paths you use in train_and_bench_real.py >>>
SISFALL_DIR = r"D:\Sentryband\raw_data\SisFall"
PPGDALIA_DIR = r"D:\Sentryband\raw_data\PPG-DaLiA\PPG_wrist_only"
MIMIC_AF_DIR = None

# How many real subjects to hold out per dataset as "new, unseen users"
N_HELD_OUT_SISFALL_SUBJECTS = 2
N_HELD_OUT_PPGDALIA_SUBJECTS = 2

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

TRUE_STATE = {
    "normal": STATE_NORMAL, "fall": STATE_POSSIBLE_FALL,
    "heart": STATE_HEART_ALERT, "combined": STATE_COMBINED_EMERGENCY,
}


def discover_sisfall_subjects_with_fall_data(sisfall_dir: str):
    """
    Returns sorted list of subject codes that have BOTH ADL (D*.txt) and
    fall (F*.txt) trials. Real SisFall data note: the elderly subject
    group (SE01-SE15) typically has ADL-only data -- no fall trials were
    recorded for them, for the participants' physical safety during
    collection. Holding out a subject with zero fall trials would make
    a "fall detection" generalization test meaningless, so this filters
    to only subjects with real data for both classes.
    """
    root = Path(sisfall_dir)
    has_fall, has_adl = set(), set()
    for f in root.rglob("*.txt"):
        m = re.search(r"S[AE]\d+", f.stem.upper())
        if not m:
            continue
        code = m.group(0)
        if f.stem.upper().startswith("F"):
            has_fall.add(code)
        elif f.stem.upper().startswith("D"):
            has_adl.add(code)
    return sorted(has_fall & has_adl)


def discover_sisfall_subjects(sisfall_dir: str):
    """Returns sorted list of ALL subject codes found in filenames,
    regardless of which classes they have data for (see
    discover_sisfall_subjects_with_fall_data for the filtered version
    used to actually choose held-out test subjects)."""
    root = Path(sisfall_dir)
    codes = set()
    for f in root.rglob("*.txt"):
        m = re.search(r"S[AE]\d+", f.stem.upper())
        if m:
            codes.add(m.group(0))
    return sorted(codes)


def discover_ppgdalia_subjects(ppgdalia_dir: str):
    """Returns sorted list of subject codes like 'S1', 'S2' found in filenames."""
    root = Path(ppgdalia_dir)
    codes = set()
    for f in list(root.rglob("S*.npz")) + list(root.rglob("S*.pkl")):
        codes.add(f.stem.upper())
    return sorted(codes)


def main():
    for label, path in [("SISFALL_DIR", SISFALL_DIR), ("PPGDALIA_DIR", PPGDALIA_DIR)]:
        if not Path(path).exists():
            print(f"ERROR: {label} = {path!r} does not exist. Edit the path "
                  "at the top of this script.")
            sys.exit(1)

    sisfall_subjects = discover_sisfall_subjects(SISFALL_DIR)
    sisfall_fall_eligible = discover_sisfall_subjects_with_fall_data(SISFALL_DIR)
    ppgdalia_subjects = discover_ppgdalia_subjects(PPGDALIA_DIR)
    print(f"Found {len(sisfall_subjects)} real SisFall subjects total; "
          f"{len(sisfall_fall_eligible)} have both ADL and fall trials "
          f"(the rest, likely the elderly 'SE' group, have ADL-only real "
          f"data -- a genuine characteristic of SisFall, not a bug).")
    print(f"Found {len(ppgdalia_subjects)} real PPG-DaLiA subjects: {ppgdalia_subjects}")

    if len(sisfall_fall_eligible) < N_HELD_OUT_SISFALL_SUBJECTS + 1:
        print("Not enough SisFall subjects with real fall data to hold any "
              "out for a meaningful trial -- reduce N_HELD_OUT_SISFALL_SUBJECTS.")
        sys.exit(1)
    if len(ppgdalia_subjects) < N_HELD_OUT_PPGDALIA_SUBJECTS + 1:
        print("Not enough distinct PPG-DaLiA subjects found -- reduce "
              "N_HELD_OUT_PPGDALIA_SUBJECTS.")
        sys.exit(1)

    held_out_sisfall = sisfall_fall_eligible[-N_HELD_OUT_SISFALL_SUBJECTS:]
    held_out_ppgdalia = ppgdalia_subjects[-N_HELD_OUT_PPGDALIA_SUBJECTS:]
    print(f"\nHolding out as 'new, unseen users': "
          f"SisFall {held_out_sisfall}, PPG-DaLiA {held_out_ppgdalia}")

    def train_sisfall_filter(stem):
        return not any(code in stem.upper() for code in held_out_sisfall)

    def train_ppgdalia_filter(stem):
        return stem.upper() not in held_out_ppgdalia

    def test_sisfall_filter(stem):
        return any(code in stem.upper() for code in held_out_sisfall)

    def test_ppgdalia_filter(stem):
        return stem.upper() in held_out_ppgdalia

    print("\nBuilding TRAIN split from all OTHER real subjects...")
    train_ds = build_real_dataset(
        SISFALL_DIR, PPGDALIA_DIR, n_per_class=100, seed=RANDOM_SEED,
        sisfall_subject_filter=train_sisfall_filter,
        ppgdalia_subject_filter=train_ppgdalia_filter,
    )

    print("Training classifiers on the training-subjects-only data...")
    fall_clf, heart_clf = train_classifiers(
        train_ds.X_accel, train_ds.y_fall, train_ds.X_ppg, train_ds.y_heart,
        seed=RANDOM_SEED,
    )

    print("Building TEST split from ONLY the held-out real subjects "
          "(never seen during training)...")
    test_ds = build_real_dataset(
        SISFALL_DIR, PPGDALIA_DIR, n_per_class=40, seed=RANDOM_SEED + 555,
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
    overall_acc = accuracy_score(y_true, y_pred)

    lines = [
        "SentryBand -- Real User Trials Report",
        "=" * 45,
        "Genuine leave-real-subjects-out test: trained on all OTHER real",
        "SisFall/PPG-DaLiA subjects, tested ONLY on subjects the model",
        "never saw during training -- an honest generalization-to-a-new-",
        "real-person measurement, replacing the old synthetic noise-",
        "perturbation stand-in (scripts/user_trials.py).",
        "",
        f"Held-out 'new user' subjects -- SisFall: {held_out_sisfall}, "
        f"PPG-DaLiA: {held_out_ppgdalia}",
        f"Accuracy on held-out, never-before-seen real subjects: {overall_acc * 100:.2f}%",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "real_user_trials_report.txt").write_text(report + "\n")
    print(f"\nReport saved to: {REPORTS_DIR / 'real_user_trials_report.txt'}")


if __name__ == "__main__":
    main()