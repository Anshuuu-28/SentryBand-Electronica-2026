"""
Full SentryBand decision pipeline -- mirrors the four architecture
layers from the deck (Slide 6) end to end:

    Sensing Layer -> Edge Compute Layer -> Decision & Fusion Layer -> Output Layer

and the four pipeline stages from Slide 7:

    Raw Sensor Window -> Feature Extraction -> Quantized Classifier -> Class Output
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .config import (
    SAMPLE_RATE_HZ, DEFAULT_FALL_THRESHOLD, DEFAULT_HEART_THRESHOLD,
)
from .features import extract_accel_features, extract_ppg_features
from .fusion import fuse


@dataclass
class PipelineResult:
    state: str
    fall_probability: float
    heart_probability: float
    fall_flag: bool
    heart_flag: bool
    latency_ms: float


class SentryBandPipeline:
    def __init__(self, fall_clf, heart_clf,
                 fall_threshold: float = DEFAULT_FALL_THRESHOLD,
                 heart_threshold: float = DEFAULT_HEART_THRESHOLD,
                 fs: float = SAMPLE_RATE_HZ):
        self.fall_clf = fall_clf
        self.heart_clf = heart_clf
        self.fall_threshold = fall_threshold
        self.heart_threshold = heart_threshold
        self.fs = fs

    def process_window(self, accel_window: np.ndarray, ppg_window: np.ndarray) -> PipelineResult:
        t0 = perf_counter()

        accel_feats = extract_accel_features(accel_window, fs=self.fs).reshape(1, -1)
        ppg_feats = extract_ppg_features(ppg_window, fs=self.fs).reshape(1, -1)

        fall_prob = float(self.fall_clf.predict_proba(accel_feats)[0, 1])
        heart_prob = float(self.heart_clf.predict_proba(ppg_feats)[0, 1])

        fall_flag = fall_prob >= self.fall_threshold
        heart_flag = heart_prob >= self.heart_threshold

        state = fuse(fall_flag, heart_flag)

        latency_ms = (perf_counter() - t0) * 1000.0

        return PipelineResult(
            state=state,
            fall_probability=fall_prob,
            heart_probability=heart_prob,
            fall_flag=fall_flag,
            heart_flag=heart_flag,
            latency_ms=latency_ms,
        )
