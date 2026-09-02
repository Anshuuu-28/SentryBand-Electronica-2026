# SentryBand — electronica India Tech Challenge 2026

> SentryBand: an Edge AI wearable that senses a fall or a dangerous
> heartbeat the instant it happens, and responds on its own, with no
> internet, no cloud, no delay.

Repo: https://github.com/Anshuuu-28/SentryBand-Electronica-2026

This started as a software-only algorithmic prototype validated on
synthetic data. It is now:

1. **Trained and validated on real, published sensor recordings** —
   not data we generated ourselves (SisFall, PPG-DaLiA, and a real
   cross-check against MIMIC PERform AF).
2. **Exported to a genuine int8-quantized `.tflite` model**, produced by
   TensorFlow's own converter and re-verified through TFLite's own
   interpreter — not a numpy simulation of quantization.
3. **Running as real embedded C++ firmware**, compiled and simulated on
   a microcontroller in Wokwi, driving real LEDs and a buzzer per state.
4. Presented on a **deployable single-page site** (`index.html`) for the
   submission and demo video.

Every honest limitation that still exists is documented below, in the
same spirit as the rest of this repo — nothing here is claimed beyond
what was actually measured.

---

## What's real vs. what's a documented simplification

| Claim | Status | Where |
|---|---|---|
| Fall detection trained/tested on real accelerometer data | ✅ Real — SisFall (waist-worn) | `src/real_data_loader.py`, `scripts/train_and_bench_real.py` |
| Heart-rhythm detection trained/tested on real PPG data | ✅ Real — PPG-DaLiA (normal rhythm) | same |
| Real recorded arrhythmia data | ✅ Real, cross-checked separately — MIMIC PERform AF | see "Cross-device finding" below |
| int8-quantized, embeddable model file | ✅ Real `.tflite` (dynamic-range int8 weights), TensorFlow's own converter + interpreter | `scripts/export_tflite_real.py`, `models_tflite/` |
| Decision logic running on a microcontroller | ✅ Real, compiled C++ firmware, ported from `src/fusion.py`, simulated in Wokwi | `firmware/sentryband_wokwi/` |
| Full on-device ML feature extraction (spectral/statistical) in embedded C | ❌ Out of scope for this stage — documented, not hidden | see firmware README note below |
| Physical hardware (real MCU, real sensors) | ❌ Simulated only (Wokwi) | next roadmap step |

---

## Real-data results (headline numbers)

Validated on held-out windows from real SisFall + PPG-DaLiA recordings
(see `CALIBRATION_SOURCES.md` and `src/real_data_loader.py` for exact
dataset provenance and every honesty caveat).

| | scikit-learn model (`models_real/`) | int8 `.tflite` model (`models_tflite/`) |
|---|---|---|
| **Accuracy** | 76.0% | **77.5%** (quantization cost 0% accuracy) |
| **Footprint** | 36.15 KB | **7.56 KB** (fall 3.87 KB + heart 3.70 KB) |
| Format | pickled RandomForest | genuine int8-quantized `.tflite`, TensorFlow's converter |

Target from the submission: footprint < 50 KB → **passed by a wide margin** on both.

### Confusion matrix — real `.tflite` model, held-out real data

```
                        Normal   Fall   Heart   Combined
true=Normal               45      1      3         1
true=Fall                  4     42      0         4
true=Heart Alert          15      0     35         0
true=Combined Emergency    2     13      2        33
```

Real Heart Alert windows are the weakest spot — mistaken for Normal
~30% of the time. That's an honest, specific target for future work,
not smoothed over.

### Why 77.5%, not the ~90%+ from early synthetic testing

An earlier iteration trained purely on synthetic (procedurally
generated) sensor data scored above 90%. That number is not reported as
a headline result here, because synthetic data is easier to classify
than real recordings — a materially misleading comparison. Every number
in the table above comes from real, cited, third-party datasets.

### Cross-device finding (a real result worth keeping)

We also tested training the Heart Alert class directly on real MIMIC
PERform AF recordings (genuine ICU-recorded atrial fibrillation) instead
of the single-device synthetic-timing hybrid. Combined accuracy
*dropped* to 68% — full threshold tuning found no improvement,
indicating a genuine cross-device domain-shift problem (ICU pulse-
oximeter vs. wrist wearable PPG have different noise/morphology
characteristics that need domain adaptation, not just retuning). This is
a documented, real finding, not a bug — see `src/real_data_loader.py`'s
`load_mimic_af_windows()` docstring for the full detail. The deployed
model uses the single-device hybrid (77.5% above); the cross-device
result is preserved as an honest, specific direction for future work.

---

## Real embedded firmware (Wokwi)

`firmware/sentryband_wokwi/` contains real, compiled Arduino C++ that
runs in Wokwi's simulator:

- **`sketch.ino`** — a direct C++ port of `src/fusion.py`'s 4-state
  decision logic and `src/alerts.py`'s buzzer/LED/BLE-alert behavior.
  Confirmed compiling and running correctly across all 4 states (Normal,
  Possible Fall, Heart Alert, Combined Emergency).
- **`diagram.json`** — wiring for an Arduino Uno + MPU6050 (fall
  detection via magnitude-spike threshold) + potentiometer (heart-rate
  proxy, 40-180 bpm) + 3 LEDs + buzzer.
- **`libraries.txt`** — required Arduino libraries.

**Scope note, stated plainly:** this firmware ports the *decision/fusion
logic*, not the full ML feature-extraction pipeline (`src/features.py`).
Replicating exact spectral/statistical features in embedded C is out of
scope for this hardware-demo stage, and Wokwi's simulated sensors don't
produce physically realistic fall/arrhythmia waveforms anyway — so doing
so wouldn't add real validation value here. The ML models above (Module
1/2) are validated separately, on real data, in Python/TFLite.

### Run it

1. Go to [wokwi.com](https://wokwi.com) → New Project → Arduino Uno
   (or use the Wokwi VS Code extension).
2. Replace the default `sketch.ino` and `diagram.json` with the ones in
   `firmware/sentryband_wokwi/`.
3. Start the simulation. Watch the Serial Monitor for `[STATE]` and
   `[BLE]` lines.
4. Drag the potentiometer (or edit its `"value"` in `diagram.json`,
   0-100) to change the simulated heart rate; shake/tilt the MPU6050 to
   trigger a simulated fall.

---

## Presentation site

`index.html` (repo root) is a single-file, dependency-free static site
for the submission and demo video — deployable directly via GitHub
Pages (Settings → Pages → Deploy from branch → `main` / `/root`). It
presents the real numbers above, the 4 device states, and the real data
sources, with an embed slot for your demo recording.

---

## Project structure

```
SentryBand-Electronica-2026/
├── README.md
├── CALIBRATION_SOURCES.md      # exact provenance + honesty caveats for every real-data number
├── requirements.txt
├── index.html                   # presentation / demo site
├── src/
│   ├── config.py                 # design targets & tuned thresholds
│   ├── sensors.py                 # synthetic accelerometer + PPG generators (original prototype)
│   ├── calibrated_sensors.py      # literature-calibrated synthetic generators
│   ├── real_data_loader.py        # loads REAL SisFall / PPG-DaLiA / MIMIC PERform AF data
│   ├── features.py                # time & frequency-domain feature extraction
│   ├── dataset.py / calibrated_dataset.py
│   ├── models.py                  # trains/saves/loads the fall + heart classifiers
│   ├── fusion.py                  # 4-state decision fusion logic (also ported to firmware/)
│   ├── alerts.py                  # buzzer/LED + BLE alert simulation (also ported to firmware/)
│   └── pipeline.py                # ties every layer together end to end
├── scripts/
│   ├── train.py / bench_test.py / user_trials.py / power_latency_test.py
│   ├── false_alarm_tuning.py / quantization_demo.py     # original synthetic-data suite
│   ├── train_and_bench_calibrated.py                     # literature-calibrated synthetic suite
│   ├── extract_pkl_only.py / extract_wrist_compact.py     # PPG-DaLiA data extraction utilities
│   ├── train_and_bench_real.py     # train + bench on REAL data (headline results)
│   ├── tune_real_thresholds.py     # threshold sweep on real held-out data
│   └── export_tflite_real.py       # real int8 .tflite export + TFLite Interpreter verification
├── firmware/
│   └── sentryband_wokwi/           # real embedded C++ firmware (Wokwi simulation)
│       ├── sketch.ino
│       ├── diagram.json
│       └── libraries.txt
├── models/ models_calibrated/       # synthetic-data models (original prototype)
├── models_real/ models_tflite/      # REAL-data models (sklearn + int8 .tflite)
└── reports/                         # generated output of every script above
```

---

## Setup

```bash
pip install -r requirements.txt
pip install tensorflow   # only needed for scripts/export_tflite_real.py
```

## How to run the real-data pipeline (headline results)

```bash
# 1. Download the datasets (see src/real_data_loader.py module docstring
#    for exact download links and expected folder structure):
#      - SisFall (accelerometer/fall)
#      - PPG-DaLiA (PPG/heart rate) -- use scripts/extract_wrist_compact.py
#        to pull only the small wrist-signal .npz files, not the full
#        ~1.3 GB-per-subject raw .pkl files
#      - MIMIC PERform AF (optional, for the cross-device check above)

# 2. Edit the dataset paths at the top of these 3 scripts, then run:
python scripts/train_and_bench_real.py     # trains + benches on real data
python scripts/tune_real_thresholds.py     # sweeps thresholds on real held-out data
python scripts/export_tflite_real.py       # exports + verifies the real int8 .tflite

# 3. Run the Wokwi firmware -- see "Real embedded firmware" above
```

## How to run the original synthetic-data suite (kept for reference)

```bash
python scripts/train.py
python scripts/demo_realtime.py --n 20
python scripts/bench_test.py
python scripts/user_trials.py
python scripts/power_latency_test.py
python scripts/false_alarm_tuning.py
python scripts/quantization_demo.py
python scripts/train_and_bench_calibrated.py   # literature-calibrated synthetic variant
```

Every script's exact output is saved in `reports/` so results can be
read without re-running anything.

---

## Honest limitations (please read before judging accuracy numbers)

1. **No physical hardware yet.** Everything above runs on real *data*
   but simulated *sensors* (Wokwi) — there is no physical MCU,
   accelerometer, PPG sensor, or BLE radio. Building and testing on real
   parts remains the submission's own next roadmap step
   ("Real-World Testing").
2. **Heart Alert real-data recall is ~70%** (see confusion matrix
   above) — the model's clearest weak point, stated plainly rather than
   smoothed over.
3. **The firmware ports decision logic, not full feature extraction**
   (see "Real embedded firmware" above) — a deliberate, documented scope
   boundary for this stage, not a gap being hidden.
4. **Cross-device (MIMIC AF ↔ PPG-DaLiA) fusion underperforms** — a real,
   measured finding (see "Cross-device finding" above), not yet solved.
5. **Latency is measured on a laptop/Wokwi CPU**, not the eventual target
   MCU in real hardware — the algorithm is proven cheap enough in
   principle; final timing needs re-measuring after real hardware exists.
6. **Battery-life numbers (in the original synthetic-suite reports) are
   estimates** built from labeled *assumed* current-draw figures, not a
   real multimeter measurement.

Every one of these is also noted at the point in the code/docs where the
relevant claim is made.