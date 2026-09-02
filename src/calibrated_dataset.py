"""
Builds a labeled dataset using the literature-calibrated generators
(src/calibrated_sensors.py) instead of src/sensors.py's plain synthetic
generators. Everything downstream (features.py, models.py, fusion.py)
is reused unchanged -- only the raw-signal generation step differs.
"""

import numpy as np

from .config import RANDOM_SEED
from .calibrated_sensors import gen_calibrated_accel_window, gen_calibrated_ppg_window
from .features import extract_accel_features, extract_ppg_features
from .dataset import Dataset, CLASS_NAMES  # reuse the same Dataset container + class list


def build_calibrated_dataset(n_per_class: int = 150, seed: int = RANDOM_SEED) -> Dataset:
    rng = np.random.default_rng(seed)

    X_accel, X_ppg, y_fall, y_heart, class_name = [], [], [], [], []

    for cls in CLASS_NAMES:
        is_fall = cls in ("fall", "combined")
        is_heart = cls in ("heart", "combined")
        for _ in range(n_per_class):
            accel_win = gen_calibrated_accel_window(is_fall, rng=rng)
            ppg_win = gen_calibrated_ppg_window(is_heart, rng=rng)

            X_accel.append(extract_accel_features(accel_win))
            X_ppg.append(extract_ppg_features(ppg_win))
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
