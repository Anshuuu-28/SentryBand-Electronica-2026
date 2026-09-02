"""
False-Alarm Reduction (Slide 13, Feasibility & Validation):
    "Tune thresholds and combined-signal logic to minimize false
    positives without missing true emergencies."

This script sweeps the fall/heart decision thresholds (see
src/pipeline.py, src/config.py) over a grid and reports, for each
threshold pair:

  - False Positive Rate (FPR): fraction of true-Normal windows that are
    incorrectly flagged as any alert state.
  - Recall: fraction of true-emergency windows (fall, heart, or combined)
    that are correctly flagged as an alert state.

and recommends the threshold pair with the lowest FPR among those that
keep recall >= 0.90, matching the deck's "without missing true
emergencies" priority.

Run from the project root (after scripts/train.py):
    python scripts/false_alarm_tuning.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.dataset import CLASS_NAMES
from src.models import load_models
from src.pipeline import SentryBandPipeline
from src.sensors import gen_accel_window, gen_ppg_window
from src.config import STATE_NORMAL

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
MIN_ACCEPTABLE_RECALL = 0.90


def main(n_per_class: int = 80, seed: int = 2024):
    fall_clf, heart_clf = load_models(MODELS_DIR)

    lines = [
        "SentryBand -- False-Alarm Reduction: Threshold Sweep",
        "=" * 55,
        f"{'fall_thr':>9s} {'heart_thr':>10s} {'FPR (normal)':>14s} {'recall (emergency)':>20s}",
    ]

    results = []
    for fall_thr in THRESHOLDS:
        for heart_thr in THRESHOLDS:
            pipeline = SentryBandPipeline(fall_clf, heart_clf,
                                           fall_threshold=fall_thr,
                                           heart_threshold=heart_thr)

            rng = np.random.default_rng(seed)

            false_positives, normal_total = 0, 0
            true_positives, emergency_total = 0, 0

            for cls in CLASS_NAMES:
                is_fall = cls in ("fall", "combined")
                is_heart = cls in ("heart", "combined")
                for _ in range(n_per_class):
                    accel_win = gen_accel_window(is_fall, rng=rng)
                    ppg_win = gen_ppg_window(is_heart, rng=rng)
                    result = pipeline.process_window(accel_win, ppg_win)

                    if cls == "normal":
                        normal_total += 1
                        if result.state != STATE_NORMAL:
                            false_positives += 1
                    else:
                        emergency_total += 1
                        if result.state != STATE_NORMAL:
                            true_positives += 1

            fpr = false_positives / max(1, normal_total)
            recall = true_positives / max(1, emergency_total)
            results.append((fall_thr, heart_thr, fpr, recall))
            lines.append(f"{fall_thr:9.2f} {heart_thr:10.2f} {fpr:14.3f} {recall:20.3f}")

    # Recommend lowest-FPR pair among those meeting the recall floor.
    acceptable = [r for r in results if r[3] >= MIN_ACCEPTABLE_RECALL]
    lines.append("")
    if acceptable:
        best = min(acceptable, key=lambda r: r[2])
        lines.append(
            f"Recommended thresholds (lowest FPR with recall >= {MIN_ACCEPTABLE_RECALL:.0%}):"
        )
        lines.append(
            f"  fall_threshold={best[0]:.2f}, heart_threshold={best[1]:.2f} "
            f"-> FPR={best[2]:.3f}, recall={best[3]:.3f}"
        )
    else:
        lines.append(
            f"No threshold pair in the sweep reached recall >= {MIN_ACCEPTABLE_RECALL:.0%}. "
            "Widen the grid or improve the underlying classifiers/features."
        )

    report = "\n".join(lines)
    print(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "false_alarm_tuning_report.txt").write_text(report + "\n")
    print(f"\nReport saved to: {REPORTS_DIR / 'false_alarm_tuning_report.txt'}")


if __name__ == "__main__":
    main()
