"""
Test SentryBand's real, trained models against YOUR OWN sensor data.

This lets anyone -- a judge, a reviewer, you -- bring a CSV of raw
accelerometer and/or PPG data and see what SentryBand's actual trained
models (models_real/, the same ones behind the reported 76-77.5%
accuracy) predict for it. No retraining, no access to the original
datasets needed -- just your CSV files and this script.

INPUT FORMAT
------------
--accel_csv: a CSV with 3 columns (x, y, z), one row per sample, in g
             (no header row, or a header row -- both are auto-detected).
             Needs at least 2 seconds of data at whatever sample rate you
             specify with --accel_hz.
--ppg_csv:   a CSV with 1 column (raw PPG/pulse signal), one row per
             sample, arbitrary units. Needs at least 2 seconds of data
             at whatever sample rate you specify with --ppg_hz.

You can supply either or both. If you only supply one, that subsystem's
result is reported and the other is treated as "not provided" (fusion
needs both to produce a Combined Emergency call, but each subsystem's
own probability is still meaningful on its own).

USAGE
-----
python scripts/predict_custom_data.py --accel_csv my_accel.csv --accel_hz 50
python scripts/predict_custom_data.py --ppg_csv my_ppg.csv --ppg_hz 64
python scripts/predict_custom_data.py --accel_csv a.csv --accel_hz 50 --ppg_csv p.csv --ppg_hz 64

Add --use_tflite to run the real int8 .tflite models (models_tflite/)
through TFLite's own interpreter instead of the sklearn models
(models_real/) -- requires `pip install tensorflow`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.features import extract_accel_features, extract_ppg_features
from src.models import load_models
from src.fusion import fuse
from src.config import SAMPLE_RATE_HZ, WINDOW_SECONDS, DEFAULT_FALL_THRESHOLD, DEFAULT_HEART_THRESHOLD

WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)
MODELS_REAL_DIR = Path(__file__).resolve().parents[1] / "models_real"
MODELS_TFLITE_DIR = Path(__file__).resolve().parents[1] / "models_tflite"


def resample(x: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    n_src = x.shape[0]
    duration = n_src / src_hz
    n_dst = int(round(duration * dst_hz))
    t_src = np.linspace(0, duration, n_src, endpoint=False)
    t_dst = np.linspace(0, duration, n_dst, endpoint=False)
    if x.ndim == 1:
        return np.interp(t_dst, t_src, x)
    return np.stack([np.interp(t_dst, t_src, x[:, c]) for c in range(x.shape[1])], axis=1)


def load_csv_numeric(path: str, expected_cols: int) -> np.ndarray:
    """Loads a CSV, auto-skipping a header row if present."""
    try:
        arr = np.loadtxt(path, delimiter=",")
    except ValueError:
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
    if arr.ndim == 1 and expected_cols == 1:
        return arr
    if arr.ndim == 1 and expected_cols == 3:
        raise ValueError(f"{path}: expected 3 columns (x,y,z), got 1-D data.")
    if arr.shape[1] != expected_cols:
        raise ValueError(f"{path}: expected {expected_cols} columns, got {arr.shape[1]}.")
    return arr


def take_middle_window(x: np.ndarray, window_samples: int) -> np.ndarray:
    """Uses the middle window_samples of the (resampled) signal -- avoids
    edge artifacts at the very start/end of a user-supplied recording."""
    n = x.shape[0]
    if n < window_samples:
        raise ValueError(
            f"Only {n} samples after resampling to {SAMPLE_RATE_HZ} Hz, but "
            f"need at least {window_samples} ({WINDOW_SECONDS}s window). "
            "Supply a longer recording."
        )
    start = (n - window_samples) // 2
    return x[start:start + window_samples]


def predict_fall(accel_csv: str, accel_hz: float, use_tflite: bool):
    raw = load_csv_numeric(accel_csv, expected_cols=3)
    resampled = resample(raw, accel_hz, SAMPLE_RATE_HZ)
    window = take_middle_window(resampled, WINDOW_SAMPLES)
    features = extract_accel_features(window, fs=SAMPLE_RATE_HZ).reshape(1, -1)

    if use_tflite:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(MODELS_TFLITE_DIR / "fall_classifier.tflite"))
        interpreter.allocate_tensors()
        in_d, out_d = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        interpreter.set_tensor(in_d["index"], features.astype(in_d["dtype"]))
        interpreter.invoke()
        proba = float(interpreter.get_tensor(out_d["index"])[0, 0])
    else:
        fall_clf, _ = load_models(MODELS_REAL_DIR)
        proba = float(fall_clf.predict_proba(features)[0, 1])
    return proba


def predict_heart(ppg_csv: str, ppg_hz: float, use_tflite: bool):
    raw = load_csv_numeric(ppg_csv, expected_cols=1)
    resampled = resample(raw, ppg_hz, SAMPLE_RATE_HZ)
    window = take_middle_window(resampled, WINDOW_SAMPLES)
    features = extract_ppg_features(window, fs=SAMPLE_RATE_HZ).reshape(1, -1)

    if use_tflite:
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(MODELS_TFLITE_DIR / "heart_classifier.tflite"))
        interpreter.allocate_tensors()
        in_d, out_d = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        interpreter.set_tensor(in_d["index"], features.astype(in_d["dtype"]))
        interpreter.invoke()
        proba = float(interpreter.get_tensor(out_d["index"])[0, 0])
    else:
        _, heart_clf = load_models(MODELS_REAL_DIR)
        proba = float(heart_clf.predict_proba(features)[0, 1])
    return proba


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--accel_csv", default=None, help="CSV of raw x,y,z accelerometer data (in g)")
    parser.add_argument("--accel_hz", type=float, default=None, help="Sample rate of --accel_csv")
    parser.add_argument("--ppg_csv", default=None, help="CSV of raw PPG signal (1 column)")
    parser.add_argument("--ppg_hz", type=float, default=None, help="Sample rate of --ppg_csv")
    parser.add_argument("--use_tflite", action="store_true",
                         help="Use the real int8 .tflite models instead of the sklearn ones")
    args = parser.parse_args()

    if not args.accel_csv and not args.ppg_csv:
        print("ERROR: supply at least one of --accel_csv or --ppg_csv (with matching --accel_hz/--ppg_hz).")
        sys.exit(1)
    if args.accel_csv and args.accel_hz is None:
        print("ERROR: --accel_csv given without --accel_hz.")
        sys.exit(1)
    if args.ppg_csv and args.ppg_hz is None:
        print("ERROR: --ppg_csv given without --ppg_hz.")
        sys.exit(1)

    print(f"Model source: {'real int8 .tflite (models_tflite/)' if args.use_tflite else 'sklearn (models_real/)'}")

    try:
        fall_proba = None
        heart_proba = None

        if args.accel_csv:
            fall_proba = predict_fall(args.accel_csv, args.accel_hz, args.use_tflite)
            print(f"\nFall probability   : {fall_proba:.3f}  "
                  f"(threshold {DEFAULT_FALL_THRESHOLD} -> "
                  f"{'POSSIBLE FALL' if fall_proba >= DEFAULT_FALL_THRESHOLD else 'no fall detected'})")

        if args.ppg_csv:
            heart_proba = predict_heart(args.ppg_csv, args.ppg_hz, args.use_tflite)
            print(f"Heart alert probability: {heart_proba:.3f}  "
                  f"(threshold {DEFAULT_HEART_THRESHOLD} -> "
                  f"{'HEART ALERT' if heart_proba >= DEFAULT_HEART_THRESHOLD else 'normal rhythm'})")
    except (ValueError, OSError) as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    if fall_proba is not None and heart_proba is not None:
        state = fuse(fall_proba >= DEFAULT_FALL_THRESHOLD, heart_proba >= DEFAULT_HEART_THRESHOLD)
        print(f"\nFinal fused state: {state}")
    else:
        print("\n(Supply BOTH --accel_csv and --ppg_csv to see the final fused 4-state result.)")


if __name__ == "__main__":
    main()