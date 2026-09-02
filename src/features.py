"""
Feature extraction.

Implements the "Feature Extraction (time & frequency domain)" stage from
the submitted deck's AI model pipeline (Slide 7):

    Raw Sensor Window -> Feature Extraction -> Quantized Classifier -> Class Output

Two extractors are provided, one per sensor named in the "Hardware &
Tech Stack" slide: the accelerometer (motion) and the optical PPG sensor
(heart rate).
"""

from typing import List

import numpy as np
from scipy.signal import find_peaks

from .config import SAMPLE_RATE_HZ

ACCEL_FEATURE_NAMES: List[str] = [
    "mag_mean", "mag_std", "mag_min", "mag_max", "mag_ptp",
    "jerk_max", "sma", "dominant_freq_hz", "zero_cross_rate",
]

PPG_FEATURE_NAMES: List[str] = [
    "est_hr_bpm", "rr_std_s", "peak_count", "amp_std",
    "dominant_freq_hz", "spectral_energy_ratio",
]


def extract_accel_features(window: np.ndarray, fs: float = SAMPLE_RATE_HZ) -> np.ndarray:
    """window: shape [n_samples, 3] in g. Returns a fixed-length feature vector."""
    mag = np.linalg.norm(window, axis=1)

    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))
    mag_min = float(np.min(mag))
    mag_max = float(np.max(mag))
    mag_ptp = float(np.ptp(mag))

    jerk = np.diff(mag) * fs
    jerk_max = float(np.max(np.abs(jerk))) if len(jerk) else 0.0

    # Signal Magnitude Area: mean of summed absolute per-axis values.
    sma = float(np.mean(np.sum(np.abs(window), axis=1)))

    # Dominant frequency via FFT of the magnitude signal.
    dominant_freq = _dominant_frequency(mag, fs)

    # Zero-crossing rate around the mean -> rough measure of oscillation.
    centered = mag - mag_mean
    zero_cross_rate = float(np.mean(np.diff(np.sign(centered)) != 0)) if len(centered) > 1 else 0.0

    return np.array([
        mag_mean, mag_std, mag_min, mag_max, mag_ptp,
        jerk_max, sma, dominant_freq, zero_cross_rate,
    ])


def extract_ppg_features(window: np.ndarray, fs: float = SAMPLE_RATE_HZ) -> np.ndarray:
    """window: shape [n_samples] arbitrary units. Returns a fixed-length feature vector."""
    window = np.asarray(window, dtype=float)

    # Minimum spacing between beats assuming a plausible max HR of ~220 bpm.
    min_distance = max(1, int(fs * 60.0 / 220.0))
    peaks, _ = find_peaks(window, distance=min_distance, prominence=0.3)

    if len(peaks) >= 2:
        rr_intervals = np.diff(peaks) / fs  # seconds
        est_hr_bpm = float(60.0 / np.mean(rr_intervals))
        rr_std = float(np.std(rr_intervals))
    else:
        # Fall back to the FFT-estimated rate if peak detection is too sparse.
        dom_freq = _dominant_frequency(window, fs)
        est_hr_bpm = float(dom_freq * 60.0)
        rr_std = 0.0

    peak_count = float(len(peaks))
    amp_std = float(np.std(window))
    dominant_freq = _dominant_frequency(window, fs)
    spectral_energy_ratio = _spectral_energy_ratio(window, fs)

    return np.array([
        est_hr_bpm, rr_std, peak_count, amp_std,
        dominant_freq, spectral_energy_ratio,
    ])


def _dominant_frequency(signal: np.ndarray, fs: float) -> float:
    signal = signal - np.mean(signal)
    if len(signal) < 4 or np.allclose(signal, 0):
        return 0.0
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(signal))
    if len(spectrum) <= 1:
        return 0.0
    # Ignore the DC bin (index 0) when finding the dominant frequency.
    dominant_idx = int(np.argmax(spectrum[1:]) + 1)
    return float(freqs[dominant_idx])


def _spectral_energy_ratio(signal: np.ndarray, fs: float,
                            band=(0.5, 3.5)) -> float:
    """Fraction of total spectral energy inside a plausible heart-rate band (0.5-3.5 Hz = 30-210 bpm)."""
    signal = signal - np.mean(signal)
    if len(signal) < 4:
        return 0.0
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2
    total = np.sum(power[1:]) + 1e-9
    band_mask = (freqs >= band[0]) & (freqs <= band[1])
    band_energy = np.sum(power[band_mask])
    return float(band_energy / total)
