"""
Literature-calibrated synthetic sensor generation.

This module does NOT contain real recorded data from SisFall or
PPG-DaLiA. It generates synthetic accelerometer and PPG windows whose
sampling rates, sensor ranges, and physiological ranges are calibrated
to match the PUBLISHED SPECIFICATIONS of those two studies, so the
prototype's numbers are grounded in real-world measurement parameters
rather than invented from general assumptions (see src/sensors.py for
the original, non-calibrated version).

This is still synthetic data. Read CALIBRATION_SOURCES.md for the full
citation list and the honest limitations of this approach before citing
these numbers anywhere.

--------------------------------------------------------------------
CITED PARAMETERS
--------------------------------------------------------------------
SisFall (Sucerquia, Lopez, Vargas-Bonilla, 2017, "SisFall: A Fall and
Movement Dataset", Sensors 17(1):198):
  - Accelerometer: Analog Devices ADXL345, configured for +-16 g, 13-bit ADC
  - Sampling rate: 200 Hz
  - Device location: WAIST (not wrist -- see limitations)
  - 38 subjects: 23 young adults (19-30 yrs), 15 elderly (60-75 yrs)
  - 19 ADL types, 15 fall types

PPG-DaLiA (Reiss, Indlekofer, Schmidt, Van Laerhoven, 2019, "Deep PPG:
Large-scale Heart Rate Estimation with Convolutional Neural Networks",
Sensors 19(14):3079; hosted at UCI ML Repository, dataset id 495):
  - Device: Empatica E4, worn on the WRIST (matches SentryBand placement)
  - PPG/BVP sampling rate: 64 Hz
  - Accelerometer sampling rate: 32 Hz
  - 15 subjects, age 21-55 (8 female, 7 male)
  - 8 daily activities: sitting, walking, cycling, stair climbing,
    driving, working, table soccer, lunch break
  - Ground-truth HR derived from chest ECG (RespiBAN device)
  - IMPORTANT: all subjects were healthy volunteers performing normal
    daily activities. PPG-DaLiA contains NO recorded cardiac arrhythmia
    events -- see limitations below.
--------------------------------------------------------------------
"""

import numpy as np
from scipy.signal import resample_poly

from .config import SAMPLE_RATE_HZ, WINDOW_SECONDS

# --- Cited native sampling rates (before resampling to the design target) ---
SISFALL_NATIVE_RATE_HZ = 200        # ADXL345 @ 200 Hz, per SisFall paper
SISFALL_ACCEL_RANGE_G = 16          # ADXL345 configured range, per SisFall paper
PPG_DALIA_PPG_RATE_HZ = 64          # Empatica E4 BVP channel, per PPG-DaLiA paper
PPG_DALIA_ACCEL_RATE_HZ = 32        # Empatica E4 accelerometer channel

# --- Cited subject/activity ranges ---
SISFALL_AGE_RANGES = {"young_adult": (19, 30), "elderly": (60, 75)}
PPG_DALIA_AGE_RANGE = (21, 55)

GRAVITY_G = 1.0


def _resample_to_target(signal: np.ndarray, native_fs: float,
                         target_fs: float = SAMPLE_RATE_HZ,
                         duration: float = WINDOW_SECONDS) -> np.ndarray:
    """Resample a native-rate signal down to the design target sample
    rate (see config.SAMPLE_RATE_HZ, the '1-25 Hz continuous' target
    named in the submitted deck's architecture slide)."""
    target_n = int(round(target_fs * duration))
    resampled = resample_poly(signal, up=int(target_fs * 10), down=int(native_fs * 10), axis=0)
    # resample_poly's output length can be off by a sample or two due to
    # rounding; trim or pad to exactly the expected window length.
    if len(resampled) >= target_n:
        return resampled[:target_n]
    pad = np.repeat(resampled[-1:], target_n - len(resampled), axis=0)
    return np.vstack([resampled, pad]) if resampled.ndim > 1 else np.concatenate([resampled, pad])


def gen_calibrated_accel_window(is_fall: bool, rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a synthetic 3-axis accelerometer window, in g, at
    SisFall's native 200 Hz, then resample down to the design target
    rate (config.SAMPLE_RATE_HZ) -- output shape matches
    src/sensors.py's gen_accel_window so it's a drop-in replacement for
    the rest of the pipeline.

    LIMITATION: SisFall's device was worn at the WAIST, not the wrist.
    Waist-worn fall signatures (single sharp torso impact) and
    wrist-worn ones (impact + potential arm-swing/bracing motion before
    impact) differ. This generator still targets a wrist placement,
    using SisFall's impact-magnitude range purely as a real-world sanity
    check on scale, not as a literal transfer of waist-worn dynamics.
    """
    if rng is None:
        rng = np.random.default_rng()

    native_n = int(SISFALL_NATIVE_RATE_HZ * WINDOW_SECONDS)
    t = np.linspace(0.0, WINDOW_SECONDS, native_n, endpoint=False)

    axis_gravity = rng.normal(loc=[0.0, 0.0, GRAVITY_G], scale=0.05, size=3)
    accel = np.tile(axis_gravity, (native_n, 1)).astype(float)

    walk_freq = rng.uniform(1.0, 2.2)
    walk_amp = rng.uniform(0.05, 0.18)
    phase = rng.uniform(0, 2 * np.pi, size=3)
    for axis in range(3):
        accel[:, axis] += walk_amp * np.sin(2 * np.pi * walk_freq * t + phase[axis])

    # ADXL345 sensor noise floor is very low (~microg/sqrt(Hz)); at this
    # abstraction level we use a small noise term consistent with a
    # clean digital accelerometer rather than a cheap analog part.
    accel += rng.normal(scale=0.03, size=accel.shape)

    if is_fall:
        impact_idx = int(rng.uniform(0.35, 0.6) * native_n)
        freefall_len = max(2, int(0.15 * native_n))
        impact_len = max(1, int(0.06 * native_n))

        ff_start = max(0, impact_idx - freefall_len)
        for axis in range(3):
            accel[ff_start:impact_idx, axis] *= rng.uniform(0.05, 0.25)

        # Impact magnitude: bounded within the ADXL345's configured
        # +-16 g range cited in the SisFall paper, and within the
        # 3-6 g order-of-magnitude typically reported for fall impacts
        # in fall-detection literature using this sensor.
        impact_mag = rng.uniform(3.0, min(8.0, SISFALL_ACCEL_RANGE_G * 0.5))
        impact_end = min(native_n, impact_idx + impact_len)
        spike_dir = rng.normal(size=3)
        spike_dir /= np.linalg.norm(spike_dir) + 1e-9
        for i in range(impact_idx, impact_end):
            accel[i, :] = spike_dir * impact_mag + rng.normal(scale=0.2, size=3)

        post_start = impact_end
        if post_start < native_n:
            still_gravity = rng.normal(loc=[0.0, 0.0, GRAVITY_G], scale=0.05, size=3)
            accel[post_start:, :] = still_gravity + rng.normal(scale=0.02, size=(native_n - post_start, 3))

    return _resample_to_target(accel, native_fs=SISFALL_NATIVE_RATE_HZ)


def gen_calibrated_ppg_window(is_heart_alert: bool, rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate a synthetic single-channel PPG window at PPG-DaLiA's
    native 64 Hz, then resample down to the design target rate.

    LIMITATION (important): PPG-DaLiA contains only HEALTHY subjects
    performing normal daily activities -- there is no recorded cardiac
    arrhythmia in that dataset. So:
      - The "Normal" resting/active heart-rate RANGE used here (roughly
        60-100 bpm resting, higher during activities like cycling) is
        grounded in PPG-DaLiA's actual reported per-activity mean HR
        figures.
      - The abnormal "Heart Alert" sub-patterns (tachycardia,
        bradycardia, arrhythmic beat-to-beat spacing) are NOT sourced
        from PPG-DaLiA -- they use standard clinical threshold
        definitions (e.g. tachycardia > 100-140 bpm, bradycardia < 60
        bpm depending on context; wider margins used here to keep
        classes clearly separable for this prototype) since no public,
        wrist-PPG, labeled-arrhythmia dataset was used.
    """
    if rng is None:
        rng = np.random.default_rng()

    native_n = int(PPG_DALIA_PPG_RATE_HZ * WINDOW_SECONDS)
    t = np.linspace(0.0, WINDOW_SECONDS, native_n, endpoint=False)

    if not is_heart_alert:
        # Resting-to-light-activity HR range consistent with PPG-DaLiA's
        # reported per-activity mean heart rates (sitting/working towards
        # the lower end, walking/light activity towards the higher end).
        hr_bpm = rng.uniform(60, 100)
        rr_jitter = rng.uniform(0.01, 0.03)
    else:
        subtype = rng.choice(["tachycardia", "bradycardia", "arrhythmia"])
        if subtype == "tachycardia":
            hr_bpm = rng.uniform(140, 180)
            rr_jitter = rng.uniform(0.01, 0.03)
        elif subtype == "bradycardia":
            hr_bpm = rng.uniform(30, 45)
            rr_jitter = rng.uniform(0.01, 0.03)
        else:
            hr_bpm = rng.uniform(60, 110)
            rr_jitter = rng.uniform(0.08, 0.18)

    beat_period = 60.0 / hr_bpm
    ppg = np.zeros(native_n)
    beat_time = rng.uniform(0, beat_period)
    pulse_width = beat_period * 0.35
    while beat_time < WINDOW_SECONDS:
        ppg += 1.0 * np.exp(-0.5 * ((t - beat_time) / (pulse_width * 0.4)) ** 2)
        ppg += 0.35 * np.exp(-0.5 * ((t - (beat_time + pulse_width * 0.55)) / (pulse_width * 0.3)) ** 2)
        beat_time += beat_period + rng.normal(scale=rr_jitter)

    ppg += rng.normal(scale=0.05, size=native_n)

    return _resample_to_target(ppg, native_fs=PPG_DALIA_PPG_RATE_HZ)
