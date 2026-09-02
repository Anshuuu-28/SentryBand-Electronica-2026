"""
Controlled User Trials (Slide 13, Feasibility & Validation):
    "Test with volunteers across walking, sitting, falling-simulation
    and exercise activities to tighten accuracy."

No real volunteers exist for this prototype yet (that's the roadmap's
NEXT phase). As a stand-in that is honestly labeled as synthetic, this
script generates several synthetic "users" -- each with a slightly
different sensor-noise / baseline profile via `user_baseline_shift` --
and reports per-user accuracy so we can see whether the classifier
generalizes across individual variation, not just one fixed profile.

Run from the project root (after scripts/train.py):
    python scripts/user_trials.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import accuracy_score

from src.dataset import build_dataset, CLASS_NAMES
from src.models import load_models
from src.pipeline import SentryBandPipeline
from src.config import (
    STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT, STATE_COMBINED_EMERGENCY,
)
from src.features import extract_accel_features, extract_ppg_features
from src.sensors import gen_accel_window, gen_ppg_window

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

TRUE_STATE = {
    "normal": STATE_NORMAL, "fall": STATE_POSSIBLE_FALL,
    "heart": STATE_HEART_ALERT, "combined": STATE_COMBINED_EMERGENCY,
}

# Baseline-shift magnitude per synthetic "user" -- 0.0 is the profile the
# models were trained on; larger values emulate a person whose resting
# motion/pulse baseline differs more from the training population.
SYNTHETIC_USERS = {
    "user_A (baseline, matches training)": 0.0,
    "user_B (mild variation)": 0.05,
    "user_C (moderate variation)": 0.10,
    "user_D (high variation)": 0.18,
}


def evaluate_user(fall_clf, heart_clf, shift: float, n_per_class: int = 60, seed: int = 7):
    pipeline = SentryBandPipeline(fall_clf, heart_clf)
    rng = np.random.default_rng(seed)

    y_true, y_pred = [], []
    for cls in CLASS_NAMES:
        is_fall = cls in ("fall", "combined")
        is_heart = cls in ("heart", "combined")
        for _ in range(n_per_class):
            accel_win = gen_accel_window(is_fall, rng=rng)
            ppg_win = gen_ppg_window(is_heart, rng=rng)
            if shift:
                accel_win = accel_win + rng.normal(scale=shift, size=accel_win.shape)
                ppg_win = ppg_win * (1.0 + rng.normal(scale=shift))
            result = pipeline.process_window(accel_win, ppg_win)
            y_true.append(TRUE_STATE[cls])
            y_pred.append(result.state)

    return accuracy_score(y_true, y_pred)


def main():
    fall_clf, heart_clf = load_models(MODELS_DIR)

    lines = [
        "SentryBand -- Controlled User Trials Report (synthetic users)",
        "=" * 62,
        "Each 'user' below perturbs sensor windows with a different",
        "baseline-noise magnitude to emulate person-to-person variation.",
        "This is a synthetic stand-in for real volunteer trials (roadmap:",
        "'Real-World Testing' phase). See README limitations.",
        "",
    ]

    accs = []
    for name, shift in SYNTHETIC_USERS.items():
        acc = evaluate_user(fall_clf, heart_clf, shift)
        accs.append(acc)
        lines.append(f"  {name:38s} : {acc * 100:6.2f}% accuracy")

    lines.append("")
    lines.append(f"Mean accuracy across synthetic users: {np.mean(accs) * 100:.2f}%")
    lines.append(f"Min accuracy (hardest user)          : {np.min(accs) * 100:.2f}%")

    report = "\n".join(lines)
    print(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "user_trials_report.txt").write_text(report + "\n")
    print(f"\nReport saved to: {REPORTS_DIR / 'user_trials_report.txt'}")


if __name__ == "__main__":
    main()
