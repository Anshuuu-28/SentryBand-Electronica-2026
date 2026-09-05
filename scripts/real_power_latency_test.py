"""
Module 2 (extension) -- Real Latency Measurement.

Replaces the synthetic-window latency measurement in
scripts/power_latency_test.py with a REAL measurement: runs your actual
int8 .tflite models (models_tflite/) through tf.lite.Interpreter over
real held-out SisFall/PPG-DaLiA windows, and reports mean/p95/max
latency for that genuine inference path.

IMPORTANT, unchanged from power_latency_test.py: the battery-life
estimate CANNOT be made real without physical hardware (a multimeter on
an actual MCU). No software change fixes that. This script keeps that
section, still clearly labeled ASSUMED, for continuity with the original
report -- only the latency number becomes a real measurement here.

Run AFTER scripts/export_tflite_real.py (needs models_tflite/ to exist):
    python scripts/real_power_latency_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.real_data_loader import build_real_dataset
from src.config import INFERENCE_LATENCY_TARGET_MS, WINDOW_SECONDS, RANDOM_SEED

SISFALL_DIR = r"D:\Sentryband\raw_data\SisFall"
PPGDALIA_DIR = r"D:\Sentryband\raw_data\PPG-DaLiA\PPG_wrist_only"
MIMIC_AF_DIR = None

MODELS_DIR = Path(__file__).resolve().parents[1] / "models_tflite"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

# Same clearly-labeled ASSUMPTIONS as power_latency_test.py -- unavoidable
# without physical hardware, kept identical for continuity.
ASSUMED_ACTIVE_CURRENT_MA = 6.0
ASSUMED_SLEEP_CURRENT_UA = 8.0
ASSUMED_BATTERY_MAH = 100.0
ASSUMED_DUTY_CYCLE_ACTIVE_S = 0.05


def measure_tflite_latency_ms(tflite_path: Path, X: np.ndarray) -> np.ndarray:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    latencies = []
    for i in range(len(X)):
        x = X[i:i + 1].astype(in_detail["dtype"])
        t0 = time.perf_counter()
        interpreter.set_tensor(in_detail["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(out_detail["index"])
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return np.array(latencies)


def main():
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError:
        print("ERROR: TensorFlow is not installed. Run: pip install tensorflow")
        sys.exit(1)

    for label, path in [("SISFALL_DIR", SISFALL_DIR), ("PPGDALIA_DIR", PPGDALIA_DIR)]:
        if not Path(path).exists():
            print(f"ERROR: {label} = {path!r} does not exist.")
            sys.exit(1)
    fall_tflite = MODELS_DIR / "fall_classifier.tflite"
    heart_tflite = MODELS_DIR / "heart_classifier.tflite"
    if not fall_tflite.exists() or not heart_tflite.exists():
        print(f"ERROR: expected {fall_tflite} and {heart_tflite} to exist. "
              "Run scripts/export_tflite_real.py first.")
        sys.exit(1)

    print("Building real held-out test data...")
    test_ds = build_real_dataset(SISFALL_DIR, PPGDALIA_DIR, n_per_class=50,
                                  seed=RANDOM_SEED + 999, mimic_af_dir=MIMIC_AF_DIR)

    print("Measuring REAL latency through the fall classifier .tflite...")
    fall_lat = measure_tflite_latency_ms(fall_tflite, test_ds.X_accel)
    print("Measuring REAL latency through the heart classifier .tflite...")
    heart_lat = measure_tflite_latency_ms(heart_tflite, test_ds.X_ppg)

    # Per decision window, both classifiers run once each (real pipeline)
    combined_lat = fall_lat + heart_lat
    mean_lat = float(np.mean(combined_lat))
    p95_lat = float(np.percentile(combined_lat, 95))
    max_lat = float(np.max(combined_lat))

    duty_cycle_fraction = ASSUMED_DUTY_CYCLE_ACTIVE_S / WINDOW_SECONDS
    avg_current_ma = (
        duty_cycle_fraction * ASSUMED_ACTIVE_CURRENT_MA
        + (1 - duty_cycle_fraction) * (ASSUMED_SLEEP_CURRENT_UA / 1000.0)
    )
    est_battery_life_hours = ASSUMED_BATTERY_MAH / avg_current_ma
    est_battery_life_days = est_battery_life_hours / 24.0

    lines = [
        "SentryBand -- Real Power & Latency Test Report",
        "=" * 48,
        "",
        f"1) REALLY MEASURED: fall+heart .tflite inference latency over "
        f"{len(combined_lat)} real held-out windows",
        f"   (via tf.lite.Interpreter on the actual models_tflite/ files, "
        f"Python-process overhead included -- real MCU latency will differ, "
        f"but this replaces the earlier synthetic-window measurement)",
        f"   mean : {mean_lat:.3f} ms",
        f"   p95  : {p95_lat:.3f} ms",
        f"   max  : {max_lat:.3f} ms",
        f"   design target (Slide 7): < {INFERENCE_LATENCY_TARGET_MS} ms per decision window",
        f"   status vs target (p95) : {'PASS' if p95_lat < INFERENCE_LATENCY_TARGET_MS else 'OVER TARGET'}",
        "",
        "2) ESTIMATED (STILL NOT measured -- no physical hardware exists yet;",
        "   this cannot become a real measurement without a multimeter on an",
        "   actual MCU, unlike the latency number above):",
        f"   assumed active current   : {ASSUMED_ACTIVE_CURRENT_MA} mA",
        f"   assumed sleep current    : {ASSUMED_SLEEP_CURRENT_UA} uA",
        f"   assumed active time/cycle: {ASSUMED_DUTY_CYCLE_ACTIVE_S * 1000:.0f} ms per {WINDOW_SECONDS:.0f}s window",
        f"   assumed battery capacity : {ASSUMED_BATTERY_MAH} mAh",
        f"   -> estimated avg current : {avg_current_ma:.3f} mA",
        f"   -> estimated battery life: {est_battery_life_hours:.1f} hours (~{est_battery_life_days:.1f} days)",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "real_power_latency_report.txt").write_text(report + "\n")
    print(f"\nReport saved to: {REPORTS_DIR / 'real_power_latency_report.txt'}")


if __name__ == "__main__":
    main()