"""
Bench Testing (Slide 13, Feasibility & Validation):
    "Validate fall and heart-rhythm classification against labeled
    motion + PPG datasets on the dev board."

Here "the dev board" is stood in for by this PC-based prototype (see
README limitations). This script builds a held-out synthetic test set
(different random seed than training), runs the full pipeline on every
window, and reports:

    - per-class accuracy + a 4-state confusion matrix
    - overall accuracy
    - average / p95 decision latency vs. the < 50 ms design target

Run from the project root (after scripts/train.py):
    python scripts/bench_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

from src.dataset import build_dataset, CLASS_NAMES
from src.sensors import gen_accel_window, gen_ppg_window
from src.models import load_models
from src.pipeline import SentryBandPipeline
from src.config import (
    STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT,
    STATE_COMBINED_EMERGENCY, INFERENCE_LATENCY_TARGET_MS,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

TRUE_STATE = {
    "normal": STATE_NORMAL,
    "fall": STATE_POSSIBLE_FALL,
    "heart": STATE_HEART_ALERT,
    "combined": STATE_COMBINED_EMERGENCY,
}
STATE_ORDER = [STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT, STATE_COMBINED_EMERGENCY]


def main(n_per_class: int = 100, seed: int = 999):
    fall_clf, heart_clf = load_models(MODELS_DIR)
    pipeline = SentryBandPipeline(fall_clf, heart_clf)

    rng = np.random.default_rng(seed)

    y_true, y_pred, latencies = [], [], []

    for cls in CLASS_NAMES:
        is_fall = cls in ("fall", "combined")
        is_heart = cls in ("heart", "combined")
        for _ in range(n_per_class):
            accel_win = gen_accel_window(is_fall, rng=rng)
            ppg_win = gen_ppg_window(is_heart, rng=rng)
            result = pipeline.process_window(accel_win, ppg_win)

            y_true.append(TRUE_STATE[cls])
            y_pred.append(result.state)
            latencies.append(result.latency_ms)

    overall_acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=STATE_ORDER)

    lat = np.array(latencies)
    mean_lat = float(np.mean(lat))
    p95_lat = float(np.percentile(lat, 95))
    max_lat = float(np.max(lat))

    lines = []
    lines.append("SentryBand -- Bench Test Report (synthetic held-out data)")
    lines.append("=" * 58)
    lines.append(f"Samples per class : {n_per_class}  (total {n_per_class * 4})")
    lines.append(f"Overall accuracy  : {overall_acc * 100:.2f}%")
    lines.append("")
    lines.append("Per-class accuracy:")
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    offset = 0
    for cls in CLASS_NAMES:
        sl = slice(offset, offset + n_per_class)
        cls_acc = accuracy_score(y_true_arr[sl], y_pred_arr[sl])
        lines.append(f"  {cls:10s} (true state: {TRUE_STATE[cls]:20s}) : {cls_acc * 100:6.2f}%")
        offset += n_per_class

    lines.append("")
    lines.append("Confusion matrix (rows = true state, cols = predicted state):")
    header = "                        " + "".join(f"{s[:12]:>14s}" for s in STATE_ORDER)
    lines.append(header)
    for i, s in enumerate(STATE_ORDER):
        row = "".join(f"{cm[i, j]:14d}" for j in range(len(STATE_ORDER)))
        lines.append(f"  true={s:20s}{row}")

    lines.append("")
    lines.append("Decision latency (feature extraction + both classifiers + fusion):")
    lines.append(f"  mean : {mean_lat:.3f} ms")
    lines.append(f"  p95  : {p95_lat:.3f} ms")
    lines.append(f"  max  : {max_lat:.3f} ms")
    lines.append(f"  design target (Slide 7): < {INFERENCE_LATENCY_TARGET_MS} ms per decision window")
    lines.append(f"  status vs target (p95) : {'PASS' if p95_lat < INFERENCE_LATENCY_TARGET_MS else 'OVER TARGET'}")
    lines.append("")
    lines.append("Note: latency is measured on this development machine's CPU, not")
    lines.append("the target microcontroller. It validates that the ALGORITHM is cheap")
    lines.append("enough in principle to fit a tight latency budget; final on-MCU timing")
    lines.append("must be re-measured once ported to embedded C, per the roadmap's")
    lines.append("'Real-World Testing' phase.")

    report = "\n".join(lines)
    print(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "bench_report.txt").write_text(report + "\n")
    print(f"\nReport saved to: {REPORTS_DIR / 'bench_report.txt'}")


if __name__ == "__main__":
    main()
