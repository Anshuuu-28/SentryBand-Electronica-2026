# Calibration Sources & Limitations

This document lists exactly where every "real-world calibrated" number in
`src/calibrated_sensors.py` comes from, and is honest about what this
approach does and does not prove.

## What this is

`src/calibrated_sensors.py` generates **synthetic** accelerometer and PPG
data — same as the original `src/sensors.py` — but the generation
parameters (sampling rates, sensor ranges, subject demographics, activity
types) are set to match the **published specifications** of two real,
peer-reviewed wearable-sensor datasets, instead of being estimated from
general domain knowledge.

## What this is NOT

This is **not** real recorded human data. No files from SisFall or
PPG-DaLiA are read, downloaded, or included anywhere in this repository.
If asked directly: *"Is this trained on real SisFall/PPG-DaLiA data?"* —
the honest answer is **no**; it is synthetic data whose generation
parameters are calibrated to match those datasets' published specs.

---

## Citations

### SisFall (used to calibrate the accelerometer/fall generator)

> Sucerquia, A., López, J.D., Vargas-Bonilla, J.F. (2017). "SisFall: A
> Fall and Movement Dataset." *Sensors*, 17(1), 198.
> https://doi.org/10.3390/s17010198

Parameters taken from this paper and used in `calibrated_sensors.py`:

| Parameter | Published value | Used in prototype |
|---|---|---|
| Accelerometer model | ADXL345 | Referenced for range only (no chip emulation) |
| Sensor range | ±16 g | Fall-impact magnitude capped at half this range (3-8g), consistent with typical reported fall-impact magnitudes in fall-detection literature using this sensor |
| Sampling rate | 200 Hz | Native generation rate before resampling to the design target (1-25 Hz, per the submitted deck) |
| Device placement | **Waist** | See limitation below — SentryBand is wrist-worn |
| Subjects | 38 (23 young adults 19-30, 15 elderly 60-75) | Age ranges referenced in docstrings only, not separately modeled as distinct subject profiles in this version |
| Activities | 19 ADL types, 15 fall types | Not individually modeled — this prototype only distinguishes "fall" vs. "not fall" at the class level, per the deck's decision table |

### PPG-DaLiA (used to calibrate the PPG/heart-rate generator)

> Reiss, A., Indlekofer, I., Schmidt, P., Van Laerhoven, K. (2019). "Deep
> PPG: Large-Scale Heart Rate Estimation with Convolutional Neural
> Networks." *Sensors*, 19(14), 3079. https://doi.org/10.3390/s19143079
> Hosted at: UCI Machine Learning Repository, dataset id 495,
> https://archive.ics.uci.edu/dataset/495/ppg+dalia

| Parameter | Published value | Used in prototype |
|---|---|---|
| Device | Empatica E4, **wrist**-worn | Matches SentryBand's wrist placement |
| PPG (BVP) sampling rate | 64 Hz | Native generation rate before resampling |
| Accelerometer sampling rate | 32 Hz | Referenced but not directly used (accel calibration comes from SisFall instead) |
| Subjects | 15 (age 21-55, 8F/7M) | Age range referenced in docstrings only |
| Activities | 8 daily activities (sitting, walking, cycling, stairs, driving, working, table soccer, lunch) | Used to justify the "Normal" class's resting-to-active heart-rate range (60-100 bpm) |
| Ground truth | ECG from chest RespiBAN | Not used — this prototype has no ECG channel |

---

## Honest limitations (please read before citing these numbers)

1. **Waist vs. wrist mismatch (SisFall).** SisFall's device was worn at
   the *waist*, not the wrist. A waist-worn fall signature (a single
   sharp torso impact) can differ from a wrist-worn one (which may also
   pick up arm-swing or a bracing motion before impact). This prototype
   uses SisFall's impact-magnitude *range* purely as a real-world sanity
   check on scale, not as a literal transfer of waist-worn dynamics to
   the wrist. A wrist-specific fall dataset (e.g. FallAllD, which
   includes a wrist sensor) would close this gap more precisely.

2. **No real arrhythmia data (PPG-DaLiA).** All 15 PPG-DaLiA subjects
   were healthy volunteers performing ordinary daily activities — the
   dataset contains no recorded cardiac arrhythmia events. This
   prototype's "Heart Alert" abnormal-rhythm patterns (tachycardia,
   bradycardia, irregular spacing) are therefore calibrated from
   **standard clinical threshold definitions**, not from PPG-DaLiA
   itself. Only the "Normal" class's realistic resting/active heart-rate
   *range* is grounded in PPG-DaLiA's reported figures.

3. **Still synthetic, still on a PC.** Exactly like the original
   prototype, this is software validation of the pipeline's logic, not a
   claim that physical hardware or real patient data has been used.

4. **No per-subject modeling.** Real datasets have meaningful
   person-to-person variation (see `scripts/user_trials.py` for a
   separate, simpler simulation of that). This calibration effort
   focused on getting the *sensor and signal parameters* right, not on
   modeling individual subject variability from either dataset.

## Why do this at all, then?

Because it is a real, verifiable improvement over pure invention: every
sampling rate, sensor range, and heart-rate figure used here can be
checked against a cited, peer-reviewed source — instead of "this seemed
like a reasonable number." It is the honest middle ground between "fully
invented synthetic data" and "downloading and processing real recordings,"
achievable at zero cost and without large dataset downloads.
