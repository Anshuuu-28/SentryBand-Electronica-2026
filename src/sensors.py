"""
Synthetic sensor generators.

SentryBand's proposed hardware (idea submission, "Hardware & Tech Stack")
is a 3-axis accelerometer plus an optical PPG heart-rate sensor. Since no
physical prototype board exists yet (see README / roadmap: "Real-World
Testing" is the NEXT phase, not the current one), this module generates
physiologically-plausible synthetic windows for each of the four states
named in the deck's decision table (Slide 9):

    Normal, Possible Fall, Heart Alert, Combined Emergency

These synthetic signals are only used to validate the pipeline's logic
end-to-end. They are not recordings from real hardware.
"""

import numpy as np

from .config import SAMPLE_RATE_HZ, WINDOW_SECONDS

GRAVITY_G = 1.0


def _time_vector(fs: float, duration: float) -> np.ndarray:
    n = int(round(fs * duration))
    return np.linspace(0.0, duration, n, endpoint=False)


def gen_accel_window(is_fall: bool, fs: float = SAMPLE_RATE_HZ,
                      duration: float = WINDOW_SECONDS,
                      rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a synthetic 3-axis accelerometer window (shape: [n_samples, 3]),
    in units of g.

    Normal: gentle walking-like oscillation around gravity + sensor noise.
    Fall:   a short low-magnitude "free-fall" dip followed by a sharp
            high-magnitude impact spike, then near-stillness -- the
            textbook fall signature used by wearable fall detectors.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = _time_vector(fs, duration)
    n = len(t)

    # Baseline: mostly still, tilted axes carry gravity components.
    axis_gravity = rng.normal(loc=[0.0, 0.0, GRAVITY_G], scale=0.05, size=3)
    accel = np.tile(axis_gravity, (n, 1)).astype(float)

    # Everyday micro-motion (walking/handling the wrist) on top of gravity.
    walk_freq = rng.uniform(1.0, 2.2)  # Hz, typical walking cadence range
    walk_amp = rng.uniform(0.05, 0.18)
    phase = rng.uniform(0, 2 * np.pi, size=3)
    for axis in range(3):
        accel[:, axis] += walk_amp * np.sin(2 * np.pi * walk_freq * t + phase[axis])

    # Sensor noise
    accel += rng.normal(scale=0.03, size=accel.shape)

    if is_fall:
        # Choose an impact location, leaving room for pre/post segments.
        impact_idx = int(rng.uniform(0.35, 0.6) * n)
        freefall_len = max(2, int(0.15 * n))
        impact_len = max(1, int(0.06 * n))

        # Free-fall dip: magnitude drops toward ~0g just before impact.
        ff_start = max(0, impact_idx - freefall_len)
        for axis in range(3):
            accel[ff_start:impact_idx, axis] *= rng.uniform(0.05, 0.25)

        # Impact spike: large, brief acceleration transient.
        impact_mag = rng.uniform(3.0, 6.0)
        impact_end = min(n, impact_idx + impact_len)
        spike_dir = rng.normal(size=3)
        spike_dir /= np.linalg.norm(spike_dir) + 1e-9
        for i in range(impact_idx, impact_end):
            accel[i, :] = spike_dir * impact_mag + rng.normal(scale=0.2, size=3)

        # Post-fall: person lying still -> very low variance.
        post_start = impact_end
        if post_start < n:
            still_gravity = rng.normal(loc=[0.0, 0.0, GRAVITY_G], scale=0.05, size=3)
            accel[post_start:, :] = still_gravity + rng.normal(scale=0.02, size=(n - post_start, 3))

    return accel


def gen_ppg_window(is_heart_alert: bool, fs: float = SAMPLE_RATE_HZ,
                    duration: float = WINDOW_SECONDS,
                    rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a synthetic single-channel PPG window (shape: [n_samples]),
    arbitrary units.

    Normal:      regular pulse waveform, heart rate in a resting-healthy
                 range (60-100 bpm), even beat-to-beat spacing.
    Heart Alert: one of three abnormal sub-patterns chosen at random --
                 tachycardia (>140 bpm), bradycardia (<45 bpm), or an
                 irregular/arrhythmic beat-to-beat spacing.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = _time_vector(fs, duration)
    n = len(t)

    if not is_heart_alert:
        hr_bpm = rng.uniform(60, 100)
        rr_jitter = rng.uniform(0.01, 0.03)  # seconds, healthy low HRV jitter
    else:
        subtype = rng.choice(["tachycardia", "bradycardia", "arrhythmia"])
        if subtype == "tachycardia":
            hr_bpm = rng.uniform(140, 180)
            rr_jitter = rng.uniform(0.01, 0.03)
        elif subtype == "bradycardia":
            hr_bpm = rng.uniform(30, 45)
            rr_jitter = rng.uniform(0.01, 0.03)
        else:  # arrhythmia: irregular spacing regardless of average rate
            hr_bpm = rng.uniform(60, 110)
            rr_jitter = rng.uniform(0.08, 0.18)

    beat_period = 60.0 / hr_bpm

    # Build the waveform by placing a pulse-shaped bump at each beat time,
    # with beat-to-beat jitter controlling regularity.
    ppg = np.zeros(n)
    beat_time = rng.uniform(0, beat_period)
    pulse_width = beat_period * 0.35
    while beat_time < duration:
        # Two-lobed pulse shape (systolic peak + smaller dicrotic notch bump)
        ppg += 1.0 * np.exp(-0.5 * ((t - beat_time) / (pulse_width * 0.4)) ** 2)
        ppg += 0.35 * np.exp(-0.5 * ((t - (beat_time + pulse_width * 0.55)) / (pulse_width * 0.3)) ** 2)
        beat_time += beat_period + rng.normal(scale=rr_jitter)

    ppg += rng.normal(scale=0.05, size=n)
    return ppg
