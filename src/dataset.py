"""
Synthetic labeled dataset generation for the four states named in the
deck's decision table (Slide 9):  Normal, Fall, Heart Alert, Combined.

Each generated sample carries two binary labels, matching the two
independent inference paths in the "Edge Compute Layer" of the
architecture (Slide 6: "Fall + Heart-Rhythm Inference Engine"):

    fall_label  : True for {fall, combined}
    heart_label : True for {heart, combined}

These are fused back into the four named states downstream by
src/fusion.py -- exactly mirroring the "Decision & Fusion Layer".
"""

from dataclasses import dataclass

import numpy as np

from .config import SAMPLE_RATE_HZ, WINDOW_SECONDS, RANDOM_SEED
from .sensors import gen_accel_window, gen_ppg_window
from .features import extract_accel_features, extract_ppg_features

CLASS_NAMES = ["normal", "fall", "heart", "combined"]


@dataclass
class Dataset:
    X_accel: np.ndarray   # [N, n_accel_features]
    X_ppg: np.ndarray     # [N, n_ppg_features]
    y_fall: np.ndarray    # [N] bool
    y_heart: np.ndarray   # [N] bool
    class_name: np.ndarray  # [N] str, one of CLASS_NAMES -- for reporting only


def build_dataset(n_per_class: int = 150, fs: float = SAMPLE_RATE_HZ,
                   duration: float = WINDOW_SECONDS, seed: int = RANDOM_SEED,
                   user_baseline_shift: float = 0.0) -> Dataset:
    """
    Generate a balanced synthetic dataset across the four classes.

    user_baseline_shift: optional small perturbation applied to make one
    synthetic "user" look slightly different from another (used by
    scripts/user_trials.py to emulate the deck's "Controlled User Trials").
    """
    rng = np.random.default_rng(seed)

    X_accel, X_ppg, y_fall, y_heart, class_name = [], [], [], [], []

    for cls in CLASS_NAMES:
        is_fall = cls in ("fall", "combined")
        is_heart = cls in ("heart", "combined")
        for _ in range(n_per_class):
            accel_win = gen_accel_window(is_fall, fs=fs, duration=duration, rng=rng)
            ppg_win = gen_ppg_window(is_heart, fs=fs, duration=duration, rng=rng)

            if user_baseline_shift:
                accel_win = accel_win + rng.normal(scale=user_baseline_shift, size=accel_win.shape)
                ppg_win = ppg_win * (1.0 + rng.normal(scale=user_baseline_shift))

            X_accel.append(extract_accel_features(accel_win, fs=fs))
            X_ppg.append(extract_ppg_features(ppg_win, fs=fs))
            y_fall.append(is_fall)
            y_heart.append(is_heart)
            class_name.append(cls)

    return Dataset(
        X_accel=np.vstack(X_accel),
        X_ppg=np.vstack(X_ppg),
        y_fall=np.array(y_fall, dtype=bool),
        y_heart=np.array(y_heart, dtype=bool),
        class_name=np.array(class_name),
    )
