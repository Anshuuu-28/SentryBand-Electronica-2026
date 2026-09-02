"""
Train and bench-test the fall/heart classifiers on the LITERATURE-
CALIBRATED synthetic dataset (src/calibrated_dataset.py), instead of
the plain synthetic dataset used by scripts/train.py.

Saves models to models_calibrated/ and a report to
reports/calibrated_bench_report.txt, kept separate from the original
models/ and reports/ so both versions remain available for comparison.

Run from the project root:
    python scripts/train_and_bench_calibrated.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

from src.calibrated_dataset import build_calibrated_dataset
from src.calibrated_sensors import (
    gen_calibrated_accel_window, gen_calibrated_ppg_window,
    SISFALL_NATIVE_RATE_HZ, SISFALL_ACCEL_RANGE_G,
    PPG_DALIA_PPG_RATE_HZ, PPG_DALIA_ACCEL_RATE_HZ,
)
from src.dataset import CLASS_NAMES
from src.models import train_classifiers, save_models, model_size_kb
from src.pipeline import SentryBandPipeline
from src.config import (
    RANDOM_SEED, MODEL_FOOTPRINT_TARGET_KB, INFERENCE_LATENCY_TARGET_MS,
    STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT, STATE_COMBINED_EMERGENCY,
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_calibrated"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

TRUE_STATE = {
    "normal": STATE_NORMAL, "fall": STATE_POSSIBLE_FALL,
    "heart": STATE_HEART_ALERT, "combined": STATE_COMBINED_EMERGENCY,
}
STATE_ORDER = [STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT, STATE_COMBINED_EMERGENCY]


def main():
    print("Building literature-calibrated training dataset (150 windows/class)...")
    train_ds = build_calibrated_dataset(n_per_class=150, seed=RANDOM_SEED)

    print("Training fall + heart classifiers on calibrated features...")
    fall_clf, heart_clf = train_classifiers(
        train_ds.X_accel, train_ds.y_fall, train_ds.X_ppg, train_ds.y_heart, seed=RANDOM_SEED
    )
    save_models(fall_clf, heart_clf, MODELS_DIR)

    fall_kb = model_size_kb(fall_clf)
    heart_kb = model_size_kb(heart_clf)
    total_kb = fall_kb + heart_kb

    print("Building held-out calibrated test set (100 windows/class, different seed)...")
    pipeline = SentryBandPipeline(fall_clf, heart_clf)
    rng = np.random.default_rng(2025)

    y_true, y_pred, latencies = [], [], []
    n_per_class = 100
    for cls in CLASS_NAMES:
        is_fall = cls in ("fall", "combined")
        is_heart = cls in ("heart", "combined")
        for _ in range(n_per_class):
            accel_win = gen_calibrated_accel_window(is_fall, rng=rng)
            ppg_win = gen_calibrated_ppg_window(is_heart, rng=rng)
            result = pipeline.process_window(accel_win, ppg_win)
            y_true.append(TRUE_STATE[cls])
            y_pred.append(result.state)
            latencies.append(result.latency_ms)

    overall_acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=STATE_ORDER)
    lat = np.array(latencies)

    lines = []
    lines.append("SentryBand -- Literature-Calibrated Bench Report")
    lines.append("=" * 52)
    lines.append("")
    lines.append("Data generation calibrated to published specs from:")
    lines.append(f"  SisFall (Sucerquia et al. 2017): ADXL345 @ {SISFALL_NATIVE_RATE_HZ} Hz, "
                  f"+-{SISFALL_ACCEL_RANGE_G}g range, waist-worn, 38 subjects")
    lines.append(f"  PPG-DaLiA (Reiss et al. 2019): Empatica E4, PPG @ {PPG_DALIA_PPG_RATE_HZ} Hz, "
                  f"accel @ {PPG_DALIA_ACCEL_RATE_HZ} Hz, wrist-worn, 15 subjects")
    lines.append("  Full citations + limitations: see CALIBRATION_SOURCES.md")
    lines.append("")
    lines.append("THIS IS STILL SYNTHETIC DATA -- not real recordings from either")
    lines.append("dataset. Only the generation PARAMETERS are grounded in the")
    lines.append("published specifications above.")
    lines.append("")

    lines.append(f"Model footprint: fall={fall_kb:.2f} KB, heart={heart_kb:.2f} KB, "
                  f"total={total_kb:.2f} KB (target < {MODEL_FOOTPRINT_TARGET_KB} KB) "
                  f"-> {'PASS' if total_kb < MODEL_FOOTPRINT_TARGET_KB else 'OVER TARGET'}")
    lines.append("")
    lines.append(f"Overall bench accuracy ({n_per_class}/class, held-out): {overall_acc * 100:.2f}%")
    lines.append("")
    lines.append("Per-class accuracy:")
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)
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
    lines.append(f"Decision latency: mean={np.mean(lat):.3f} ms, p95={np.percentile(lat, 95):.3f} ms "
                  f"(target < {INFERENCE_LATENCY_TARGET_MS} ms) "
                  f"-> {'PASS' if np.percentile(lat, 95) < INFERENCE_LATENCY_TARGET_MS else 'OVER TARGET'}")

    report = "\n".join(lines)
    print("\n" + report)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "calibrated_bench_report.txt").write_text(report + "\n")
    print(f"\nModels saved to : {MODELS_DIR}")
    print(f"Report saved to : {REPORTS_DIR / 'calibrated_bench_report.txt'}")


if __name__ == "__main__":
    main()
