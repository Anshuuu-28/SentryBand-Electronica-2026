"""
Train the fall and heart-rhythm classifiers on synthetic data and save
them to models/. Reports model footprint against the deck's design
target of < 50 KB (quantized) -- see src/config.py.

Run from the project root:
    python scripts/train.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import build_dataset
from src.models import train_classifiers, save_models, model_size_kb
from src.config import MODEL_FOOTPRINT_TARGET_KB, RANDOM_SEED

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def main():
    print("Building synthetic training dataset (150 windows/class)...")
    ds = build_dataset(n_per_class=150, seed=RANDOM_SEED)

    print("Training fall classifier (accelerometer features) "
          "and heart classifier (PPG features)...")
    fall_clf, heart_clf = train_classifiers(
        ds.X_accel, ds.y_fall, ds.X_ppg, ds.y_heart, seed=RANDOM_SEED
    )

    save_models(fall_clf, heart_clf, MODELS_DIR)

    fall_kb = model_size_kb(fall_clf)
    heart_kb = model_size_kb(heart_clf)
    total_kb = fall_kb + heart_kb

    lines = [
        "SentryBand -- Model Footprint Report",
        "=" * 40,
        f"Fall classifier size  : {fall_kb:7.2f} KB",
        f"Heart classifier size : {heart_kb:7.2f} KB",
        f"Combined size         : {total_kb:7.2f} KB",
        f"Design target         : < {MODEL_FOOTPRINT_TARGET_KB} KB (quantized, Slide 7)",
        f"Status                : {'PASS' if total_kb < MODEL_FOOTPRINT_TARGET_KB else 'OVER TARGET'}",
        "",
        "Note: this reports the size of the prototype's scikit-learn",
        "RandomForest objects (pickled), used to validate the fall/heart",
        "detection LOGIC on a PC. It is a stand-in for the int8-quantized,",
        "microcontroller-deployed model described in the submission -- see",
        "scripts/quantization_demo.py for an actual int8 weight-quantization",
        "demonstration, and README.md for the full limitations statement.",
    ]
    report = "\n".join(lines)
    print("\n" + report)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "model_footprint.txt").write_text(report + "\n")
    print(f"\nModels saved to: {MODELS_DIR}")
    print(f"Report saved to: {REPORTS_DIR / 'model_footprint.txt'}")


if __name__ == "__main__":
    main()
