"""
Edge Compute Layer: classifiers.

The deck (Slide 6, EDGE COMPUTE LAYER) names:
    "Quantized TinyML Model (int8, on-chip)"
    "Fall + Heart-Rhythm Inference Engine"

and Slide 7 names the model family as:
    "Quantized Classifier (int8 CNN / decision-tree ensemble)"

This module implements the "decision-tree ensemble" half of that stated
option directly (small RandomForest classifiers -- one for the fall
signal, one for the heart-rhythm signal), since it runs with no extra
dependencies. The "int8 CNN" half is demonstrated separately in
scripts/quantization_demo.py, which trains a tiny neural net and
manually int8-quantizes its weights to prove the footprint/accuracy
trade-off described in the deck without requiring TensorFlow.
"""

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config import RANDOM_SEED

# Small ensembles: enough trees to be stable, shallow + few estimators to
# keep footprint low, in line with the "< 50 KB quantized" design target.
N_ESTIMATORS = 8
MAX_DEPTH = 4


def train_classifiers(X_accel: np.ndarray, y_fall: np.ndarray,
                       X_ppg: np.ndarray, y_heart: np.ndarray,
                       seed: int = RANDOM_SEED
                       ) -> Tuple[RandomForestClassifier, RandomForestClassifier]:
    fall_clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=seed
    )
    fall_clf.fit(X_accel, y_fall)

    heart_clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=seed
    )
    heart_clf.fit(X_ppg, y_heart)

    return fall_clf, heart_clf


def model_size_kb(model) -> float:
    return len(pickle.dumps(model)) / 1024.0


def save_models(fall_clf, heart_clf, out_dir: str) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fall_classifier.pkl", "wb") as f:
        pickle.dump(fall_clf, f)
    with open(out_dir / "heart_classifier.pkl", "wb") as f:
        pickle.dump(heart_clf, f)


def load_models(in_dir: str):
    in_dir = Path(in_dir)
    with open(in_dir / "fall_classifier.pkl", "rb") as f:
        fall_clf = pickle.load(f)
    with open(in_dir / "heart_classifier.pkl", "rb") as f:
        heart_clf = pickle.load(f)
    return fall_clf, heart_clf
