"""
Module 1 — Real Sensor Data.

Loads REAL recordings from two published, freely-downloadable datasets and
reshapes them into the exact same `Dataset` object that `src/dataset.py`
(synthetic) produces, so every downstream script (train.py, bench_test.py,
user_trials.py, false_alarm_tuning.py, quantization_demo.py) works
UNCHANGED — you just point them at `build_real_dataset()` instead of
`build_dataset()`.

You do not need to write any parsing code yourself. You only need to:
  1. Download the two datasets (see DOWNLOAD INSTRUCTIONS below)
  2. Point SISFALL_DIR / PPGDALIA_DIR at wherever you extracted them
  3. Run `python scripts/train_and_bench_real.py` (added alongside this file)

--------------------------------------------------------------------------
DOWNLOAD INSTRUCTIONS
--------------------------------------------------------------------------

SisFall (real accelerometer fall + ADL recordings, waist-worn):
  The original university host (sistemic.udea.edu.co) is frequently down.
  Easiest free mirror: Kaggle, search "SisFall" (e.g. the "SisFall Enhanced"
  or original SisFall dataset uploads) — download and unzip.
  You should end up with a folder structure like:
      SisFall_dataset/
        SA01/  D01_SA01_R01.txt  D02_SA01_R01.txt  ...  F01_SA01_R01.txt ...
        SA02/  ...
        SE01/  ...
  Filenames starting with "F" = a fall trial. Filenames starting with "D"
  = an activity-of-daily-living (ADL, i.e. NOT a fall) trial.
  Each .txt file has one row per sample, 9 comma-separated columns:
      ADXL345:  ax, ay, az   (16-bit signed raw counts, range selectable,
                              SisFall's own README says +-16g, 13-bit
                              resolution -> scale factor below)
      ITG3200:  gx, gy, gz   (gyroscope -- unused here)
      MMA8451Q: ax, ay, az   (second accelerometer -- unused here, we use
                              the ADXL345 columns as the "wrist" signal)

PPG-DaLiA (real wrist PPG + accelerometer, 15 subjects):
  https://archive.ics.uci.edu/dataset/495/ppg+dalia  -> "Download" (2.7 GB
  data.zip). Unzip it. You should end up with:
      PPG_FieldStudy/
        S1/ S1.pkl
        S2/ S2.pkl
        ...
  Each Si.pkl is a Python pickle (protocol 2, load with encoding="latin1")
  containing a dict with keys:
      data['signal']['wrist']['BVP']   -> PPG-like signal @ 64 Hz (Empatica E4)
      data['signal']['wrist']['ACC']   -> 3-axis accel @ 32 Hz
      data['label']                    -> ground-truth HR (bpm) @ 0.5 Hz (every 2s),
                                           derived from chest ECG
  IMPORTANT (documented limitation, see CALIBRATION_SOURCES.md): PPG-DaLiA
  has NO arrhythmia / heart-alert-labeled data — all 15 subjects are
  healthy volunteers doing normal daily activities. So real "Heart Alert"
  windows do not exist in this dataset. This loader therefore:
    - uses REAL PPG-DaLiA windows for the "normal" heart class (real data,
      real morphology, real resting-to-active HR range)
    - synthesizes "heart alert" windows by resampling REAL beat morphology
      extracted from PPG-DaLiA but re-timing it to tachycardia/bradycardia/
      arrhythmia rates (documented as `_synthetic_heart_alert_from_real_beat`
      below) -- this is a real-morphology / synthetic-timing hybrid, NOT
      claimed as "real recorded arrhythmia data". If you later find a real
      arrhythmia PPG dataset (e.g. a subset of MIMIC-III/PhysioNet with
      PPG + arrhythmia labels) point PPGDALIA_DIR-style loading at it
      instead and this last honesty caveat goes away.
"""

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np

from .config import SAMPLE_RATE_HZ, WINDOW_SECONDS, RANDOM_SEED
from .features import extract_accel_features, extract_ppg_features
from .dataset import Dataset, CLASS_NAMES

WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)

# ---- SisFall raw-file constants (from the SisFall paper / README) ----
SISFALL_ADXL_RESOLUTION_BITS = 13
SISFALL_ADXL_RANGE_G = 16.0
SISFALL_NATIVE_HZ = 200.0


def _sisfall_counts_to_g(raw_counts: np.ndarray) -> np.ndarray:
    """Convert SisFall raw ADXL345 counts to g, per the dataset's own
    documented formula: g = (2 * range / 2^resolution) * raw_count."""
    scale = (2.0 * SISFALL_ADXL_RANGE_G) / (2 ** SISFALL_ADXL_RESOLUTION_BITS)
    return raw_counts.astype(float) * scale


def _resample(x: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """Simple linear-interpolation resample (no scipy dependency)."""
    n_src = x.shape[0]
    duration = n_src / src_hz
    n_dst = int(round(duration * dst_hz))
    t_src = np.linspace(0, duration, n_src, endpoint=False)
    t_dst = np.linspace(0, duration, n_dst, endpoint=False)
    if x.ndim == 1:
        return np.interp(t_dst, t_src, x)
    return np.stack([np.interp(t_dst, t_src, x[:, c]) for c in range(x.shape[1])], axis=1)


def _sliding_windows(x: np.ndarray, window_samples: int, hop_samples: int):
    n = x.shape[0]
    i = 0
    while i + window_samples <= n:
        yield x[i:i + window_samples]
        i += hop_samples


# --------------------------------------------------------------------------
# SisFall -> accel windows (fall / not-fall)
# --------------------------------------------------------------------------


def _read_sisfall_txt(path: Path) -> np.ndarray:
    """
    Robust line-by-line parser for SisFall's raw .txt format, which uses
    comma-separated values with irregular spacing and a trailing ';' at
    the end of each line (and sometimes trailing blank lines) -- both of
    which break a naive np.loadtxt call. Returns an [n_valid_rows, 3]
    array of the first 3 columns (ADXL345 x, y, z raw counts). Malformed
    lines are silently skipped rather than aborting the whole file.
    """
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip().rstrip(";").strip()
            if not line:
                continue
            fields = [f.strip() for f in line.split(",")]
            if len(fields) < 3:
                continue
            try:
                x, y, z = float(fields[0]), float(fields[1]), float(fields[2])
            except ValueError:
                continue
            rows.append((x, y, z))
    return np.array(rows, dtype=float)


def load_sisfall_accel_windows(sisfall_dir: str, max_files: int = None,
                                subject_filter=None):
    """
    Returns (fall_windows, adl_windows), each a list of [WINDOW_SAMPLES, 3]
    arrays in g, resampled to SAMPLE_RATE_HZ, drawn from real SisFall trials.

    subject_filter: optional callable(filename_stem) -> bool. If given,
    only files where this returns True are used -- e.g. to hold out a
    specific real subject for a leave-one-subject-out generalization test
    (see scripts/real_user_trials.py). Filenames look like
    "F01_SA01_R01.txt" / "D01_SA01_R01.txt", so subject_filter typically
    checks for a "SA01"-style substring.
    """
    root = Path(sisfall_dir)
    files = sorted(root.rglob("*.txt"))
    if subject_filter:
        files = [f for f in files if subject_filter(f.stem)]
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(
            f"No .txt files found under {sisfall_dir}"
            + (" matching subject_filter" if subject_filter else "")
            + ". Check SISFALL_DIR points at the extracted SisFall_dataset/ folder."
        )

    fall_windows, adl_windows = [], []
    for f in files:
        name = f.name.upper()
        is_fall = name.startswith("F")
        is_adl = name.startswith("D")
        if not (is_fall or is_adl):
            continue

        raw = _read_sisfall_txt(f)
        if raw.shape[0] < 10:
            continue  # unreadable / too-short file, skip rather than crash
        accel_g = _sisfall_counts_to_g(raw)
        accel_g = _resample(accel_g, SISFALL_NATIVE_HZ, SAMPLE_RATE_HZ)

        # For falls, take windows centered on the highest-magnitude sample
        # (the impact) rather than a blind sliding window, so each fall
        # trial contributes a genuinely fall-containing window.
        if is_fall:
            mag = np.linalg.norm(accel_g, axis=1)
            peak = int(np.argmax(mag))
            start = max(0, peak - WINDOW_SAMPLES // 2)
            end = start + WINDOW_SAMPLES
            if end > len(accel_g):
                start = max(0, len(accel_g) - WINDOW_SAMPLES)
                end = start + WINDOW_SAMPLES
            if end <= len(accel_g):
                fall_windows.append(accel_g[start:end])
        else:
            hop = WINDOW_SAMPLES  # non-overlapping ADL windows
            for w in _sliding_windows(accel_g, WINDOW_SAMPLES, hop):
                adl_windows.append(w)

    return fall_windows, adl_windows


# --------------------------------------------------------------------------
# PPG-DaLiA -> PPG windows (normal heart rhythm, real)
# --------------------------------------------------------------------------

def load_ppgdalia_normal_windows(ppgdalia_dir: str, max_subjects: int = None,
                                  subject_filter=None):
    """
    Returns a list of [WINDOW_SAMPLES] real PPG windows (resampled to
    SAMPLE_RATE_HZ) drawn from real PPG-DaLiA BVP signal, representing the
    "normal" heart class (these subjects have no diagnosed arrhythmia).

    Reads EITHER:
      - compact .npz files produced by scripts/extract_wrist_compact.py
        (a few MB each -- recommended, avoids the ~1.3 GB per-subject
        full .pkl footprint), OR
      - the original raw S*.pkl files (if you extracted those instead)
    Auto-detects whichever is present under ppgdalia_dir.

    subject_filter: optional callable(filename_stem) -> bool, e.g. to
    hold out a specific real subject ("S3") for a leave-one-subject-out
    generalization test (see scripts/real_user_trials.py).
    """
    root = Path(ppgdalia_dir)
    npz_files = sorted(root.rglob("S*.npz"))
    pkl_files = sorted(root.rglob("S*.pkl"))
    files = npz_files if npz_files else pkl_files
    is_npz = bool(npz_files)
    if subject_filter:
        files = [f for f in files if subject_filter(f.stem)]
    if max_subjects:
        files = files[:max_subjects]
    if not files:
        raise FileNotFoundError(
            f"No S*.npz or S*.pkl files found under {ppgdalia_dir}. Check "
            "PPGDALIA_DIR points at the folder produced by "
            "extract_wrist_compact.py (or the extracted PPG_FieldStudy/ "
            "folder if you used raw .pkl files instead)."
        )

    windows = []
    for f in files:
        if is_npz:
            with np.load(f) as npz:
                bvp = np.asarray(npz["bvp"]).reshape(-1)
        else:
            with open(f, "rb") as fh:
                data = pickle.load(fh, encoding="latin1")
            bvp = np.asarray(data["signal"]["wrist"]["BVP"]).reshape(-1)
        bvp_native_hz = 64.0
        bvp = _resample(bvp, bvp_native_hz, SAMPLE_RATE_HZ)
        # z-normalize per-subject so amplitude differences between subjects
        # don't dominate the feature scale (BVP is in arbitrary sensor units)
        bvp = (bvp - np.mean(bvp)) / (np.std(bvp) + 1e-9)

        hop = WINDOW_SAMPLES * 2  # skip-sample to keep dataset size sane
        for w in _sliding_windows(bvp, WINDOW_SAMPLES, hop):
            windows.append(w)
    return windows


def _load_mimic_af_csv_file(path: Path):
    """
    Best-effort column auto-detection for MIMIC PERform AF Dataset CSV
    files (PPG-beats CSV export, Zenodo 6967256). The exact column naming
    has varied slightly across dataset versions, so this looks for a
    column literally named one of the common variants; if none match, it
    falls back to "second numeric column" (first is usually a time axis).
    Returns a 1-D PPG array at the dataset's native 125 Hz, or None if the
    file couldn't be parsed as expected.
    """
    import csv
    with open(path, "r", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows = list(reader)
    if not header or not rows:
        return None

    header_lower = [h.strip().lower() for h in header]
    ppg_col = None
    for candidate in ("ppg", "pleth", "pulse", "signal"):
        if candidate in header_lower:
            ppg_col = header_lower.index(candidate)
            break
    if ppg_col is None and len(header_lower) >= 2:
        ppg_col = 1  # assume col 0 = time, col 1 = first signal (commonly PPG)

    if ppg_col is None:
        return None
    try:
        return np.array([float(r[ppg_col]) for r in rows if len(r) > ppg_col])
    except ValueError:
        return None


def load_mimic_af_windows(mimic_af_dir: str, native_hz: float = 125.0):
    """
    Returns (af_windows, non_af_windows): REAL PPG windows from the MIMIC
    PERform AF Dataset (Zenodo 6967256, mimic_perform_af_csv.zip),
    resampled to SAMPLE_RATE_HZ. Filenames are expected to indicate AF vs
    non-AF (the dataset ships them in separate af/ and non_af/ -style
    folders/zips) -- this function treats any path containing "non" as
    non-AF and everything else as AF; if your extracted layout differs,
    adjust the `is_af` check below (or tell me your folder names and I'll
    fix this in one edit).
    """
    root = Path(mimic_af_dir)
    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No .csv files found under {mimic_af_dir}. Check MIMIC_AF_DIR "
            "points at the extracted mimic_perform_af_csv folder."
        )

    af_windows, non_af_windows = [], []
    for f in csv_files:
        is_af = "non" not in f.as_posix().lower()
        ppg = _load_mimic_af_csv_file(f)
        if ppg is None or len(ppg) < native_hz * WINDOW_SECONDS:
            continue
        ppg = _resample(ppg, native_hz, SAMPLE_RATE_HZ)
        ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-9)
        hop = WINDOW_SAMPLES * 4
        target = af_windows if is_af else non_af_windows
        for w in _sliding_windows(ppg, WINDOW_SAMPLES, hop):
            target.append(w)

    return af_windows, non_af_windows


def _synthetic_heart_alert_from_real_beat(real_normal_window: np.ndarray,
                                            fs: float, rng: np.random.Generator) -> np.ndarray:
    """
    Builds a "heart alert" window by re-timing (NOT re-synthesizing from
    scratch) a beat extracted from a REAL PPG-DaLiA normal window. This is
    the documented real-morphology / synthetic-timing hybrid described in
    the module docstring above -- it is more honest than pure synthetic
    generation (the beat *shape* is real) but is still not real recorded
    arrhythmia data.
    """
    duration = len(real_normal_window) / fs
    t = np.linspace(0, duration, len(real_normal_window), endpoint=False)

    # crude beat-period estimate from the real window via autocorrelation
    ac = np.correlate(real_normal_window, real_normal_window, mode="full")
    ac = ac[len(ac) // 2:]
    ac[:int(0.25 * fs)] = -np.inf  # ignore implausibly short periods
    lag = int(np.argmax(ac[:int(2.0 * fs)])) or int(fs * 0.8)
    beat_period_real = max(lag / fs, 0.35)
    template_len = min(len(real_normal_window), int(beat_period_real * fs))
    template = real_normal_window[:template_len]

    subtype = rng.choice(["tachycardia", "bradycardia", "arrhythmia"])
    if subtype == "tachycardia":
        hr_bpm = rng.uniform(140, 180)
        jitter = rng.uniform(0.01, 0.03)
    elif subtype == "bradycardia":
        hr_bpm = rng.uniform(30, 45)
        jitter = rng.uniform(0.01, 0.03)
    else:
        hr_bpm = rng.uniform(60, 110)
        jitter = rng.uniform(0.08, 0.18)

    period = 60.0 / hr_bpm
    out = np.zeros(len(real_normal_window))
    beat_time = rng.uniform(0, period)
    while beat_time < duration:
        idx0 = int(beat_time * fs)
        idx1 = min(len(out), idx0 + len(template))
        if idx0 < len(out):
            out[idx0:idx1] += template[: idx1 - idx0]
        beat_time += period + rng.normal(scale=jitter)
    out += rng.normal(scale=0.05, size=out.shape)
    return out


# --------------------------------------------------------------------------
# Public entry point -- matches src/dataset.py's Dataset interface
# --------------------------------------------------------------------------

def build_real_dataset(sisfall_dir: str, ppgdalia_dir: str,
                        n_per_class: int = 100, seed: int = RANDOM_SEED,
                        max_sisfall_files: int = None,
                        max_ppgdalia_subjects: int = None,
                        mimic_af_dir: str = None,
                        sisfall_subject_filter=None,
                        ppgdalia_subject_filter=None) -> Dataset:
    """
    Builds a balanced 4-class Dataset (normal / fall / heart / combined)
    the same shape as dataset.build_dataset(), but sourced from real
    SisFall accelerometer data and real PPG data wherever a real signal
    exists for that class, per the honesty caveats in this file's module
    docstring and in CALIBRATION_SOURCES.md.

    fall/combined accel windows -> real SisFall fall trials
    normal/heart accel windows  -> real SisFall ADL trials
    normal/fall PPG windows     -> real PPG-DaLiA BVP (no cardiac event
                                    during a fall in this dataset, so fall
                                    windows reuse real resting-normal PPG)
    heart/combined PPG windows  -> REAL MIMIC PERform AF windows if
                                    mimic_af_dir is given (genuine recorded
                                    atrial fibrillation, Zenodo 6967256);
                                    otherwise falls back to the real-beat-
                                    morphology / synthetic-timing hybrid
                                    (see _synthetic_heart_alert_from_real_beat)
    """
    rng = np.random.default_rng(seed)

    fall_accel_pool, adl_accel_pool = load_sisfall_accel_windows(
        sisfall_dir, max_files=max_sisfall_files, subject_filter=sisfall_subject_filter)
    normal_ppg_pool = load_ppgdalia_normal_windows(
        ppgdalia_dir, max_subjects=max_ppgdalia_subjects, subject_filter=ppgdalia_subject_filter)

    real_af_pool = None
    if mimic_af_dir:
        af_windows, _non_af = load_mimic_af_windows(mimic_af_dir)
        if len(af_windows) == 0:
            print("  [warning] mimic_af_dir given but 0 AF windows parsed; "
                  "falling back to the synthetic-timing hybrid for Heart Alert.")
        else:
            real_af_pool = af_windows

    if len(fall_accel_pool) == 0:
        raise ValueError(
            f"Found 0 real SisFall fall windows under {sisfall_dir}. Check "
            "the folder actually contains F*.txt trial files."
        )
    if len(normal_ppg_pool) == 0:
        raise ValueError(
            f"Found 0 real PPG-DaLiA windows under {ppgdalia_dir}. Check "
            "the folder actually contains S*.pkl subject files."
        )
    if len(fall_accel_pool) < n_per_class:
        print(f"  [warning] only {len(fall_accel_pool)} real SisFall fall "
              f"windows found; sampling with replacement to reach {n_per_class}. "
              "For a real submission, use the full dataset (raise max_sisfall_files).")
    if len(normal_ppg_pool) < n_per_class:
        print(f"  [warning] only {len(normal_ppg_pool)} real PPG-DaLiA normal "
              f"windows found; sampling with replacement to reach {n_per_class}.")

    def sample(pool, k):
        idx = rng.choice(len(pool), size=k, replace=len(pool) < k)
        return [pool[i] for i in idx]

    X_accel, X_ppg, y_fall, y_heart, class_name = [], [], [], [], []

    for cls in CLASS_NAMES:
        is_fall = cls in ("fall", "combined")
        is_heart = cls in ("heart", "combined")

        accel_pool = fall_accel_pool if is_fall else adl_accel_pool
        accel_windows = sample(accel_pool, n_per_class)

        if is_heart:
            if real_af_pool is not None and len(real_af_pool) > 0:
                ppg_windows = sample(real_af_pool, n_per_class)
            else:
                base_windows = sample(normal_ppg_pool, n_per_class)
                ppg_windows = [
                    _synthetic_heart_alert_from_real_beat(w, SAMPLE_RATE_HZ, rng)
                    for w in base_windows
                ]
        else:
            ppg_windows = sample(normal_ppg_pool, n_per_class)

        for accel_w, ppg_w in zip(accel_windows, ppg_windows):
            X_accel.append(extract_accel_features(accel_w, fs=SAMPLE_RATE_HZ))
            X_ppg.append(extract_ppg_features(ppg_w, fs=SAMPLE_RATE_HZ))
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