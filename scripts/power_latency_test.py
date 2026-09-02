"""
Power & Endurance Testing (Slide 13, Feasibility & Validation):
    "Measure real current draw across sensing + inference cycles to
    confirm multi-day battery life targets."

IMPORTANT HONESTY NOTE: real current-draw measurement requires physical
hardware (a multimeter/power profiler on an actual MCU + sensors), which
does not exist yet for this prototype. This script does two things it
CAN honestly do on a PC:

  1. Measure actual decision latency (mean/p95/max) over many windows,
     the one part of the "< 50 ms" and duty-cycle story that IS
     measurable in software.
  2. Compute an ESTIMATED battery-life projection using clearly labeled
     ASSUMED current-draw figures (typical published ranges for
     Cortex-M class MCUs + low-power accel/PPG sensors in duty-cycled
     operation) -- explicitly flagged as an estimate to be replaced by
     real measurements in the next roadmap phase.

Run from the project root (after scripts/train.py):
    python scripts/power_latency_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.sensors import gen_accel_window, gen_ppg_window
from src.models import load_models
from src.pipeline import SentryBandPipeline
from src.config import INFERENCE_LATENCY_TARGET_MS, WINDOW_SECONDS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

N_TRIALS = 500

# --- Clearly-labeled ASSUMPTIONS, not measurements ---
# Typical published order-of-magnitude figures for a Cortex-M class MCU
# + low-power 3-axis accel + optical PPG sensor, used only to produce an
# illustrative estimate. MUST be replaced with real measurements on the
# actual chosen parts once hardware is available.
ASSUMED_ACTIVE_CURRENT_MA = 6.0      # sensors + MCU awake & running inference
ASSUMED_SLEEP_CURRENT_UA = 8.0       # MCU deep sleep between duty cycles
ASSUMED_BATTERY_MAH = 100.0          # small coin/LiPo cell typical of a wristband
ASSUMED_DUTY_CYCLE_ACTIVE_S = 0.05   # active time per decision window (assumed)


def main():
    fall_clf, heart_clf = load_models(MODELS_DIR)
    pipeline = SentryBandPipeline(fall_clf, heart_clf)

    rng = np.random.default_rng(123)
    latencies = []
    for _ in range(N_TRIALS):
        is_fall = bool(rng.integers(0, 2))
        is_heart = bool(rng.integers(0, 2))
        accel_win = gen_accel_window(is_fall, rng=rng)
        ppg_win = gen_ppg_window(is_heart, rng=rng)
        result = pipeline.process_window(accel_win, ppg_win)
        latencies.append(result.latency_ms)

    lat = np.array(latencies)
    mean_lat, p95_lat, max_lat = float(np.mean(lat)), float(np.percentile(lat, 95)), float(np.max(lat))

    # Duty cycle: one decision window every WINDOW_SECONDS, of which
    # ASSUMED_DUTY_CYCLE_ACTIVE_S is spent "active" (sampling + inference).
    duty_cycle_fraction = ASSUMED_DUTY_CYCLE_ACTIVE_S / WINDOW_SECONDS
    avg_current_ma = (
        duty_cycle_fraction * ASSUMED_ACTIVE_CURRENT_MA
        + (1 - duty_cycle_fraction) * (ASSUMED_SLEEP_CURRENT_UA / 1000.0)
    )
    est_battery_life_hours = ASSUMED_BATTERY_MAH / avg_current_ma
    est_battery_life_days = est_battery_life_hours / 24.0

    lines = [
        "SentryBand -- Power & Latency Test Report",
        "=" * 46,
        "",
        "1) MEASURED: decision latency over {} synthetic trials".format(N_TRIALS),
        f"   mean : {mean_lat:.3f} ms",
        f"   p95  : {p95_lat:.3f} ms",
        f"   max  : {max_lat:.3f} ms",
        f"   design target (Slide 7): < {INFERENCE_LATENCY_TARGET_MS} ms per decision window",
        f"   status vs target (p95) : {'PASS' if p95_lat < INFERENCE_LATENCY_TARGET_MS else 'OVER TARGET'}",
        "",
        "2) ESTIMATED (NOT measured -- no physical hardware yet):",
        "   battery-life projection using assumed current-draw figures.",
        f"   assumed active current   : {ASSUMED_ACTIVE_CURRENT_MA} mA",
        f"   assumed sleep current    : {ASSUMED_SLEEP_CURRENT_UA} uA",
        f"   assumed active time/cycle: {ASSUMED_DUTY_CYCLE_ACTIVE_S * 1000:.0f} ms per {WINDOW_SECONDS:.0f}s window",
        f"   assumed battery capacity : {ASSUMED_BATTERY_MAH} mAh",
        f"   -> estimated avg current : {avg_current_ma:.3f} mA",
        f"   -> estimated battery life: {est_battery_life_hours:.1f} hours (~{est_battery_life_days:.1f} days)",
        "",
        "This estimate exists only to sanity-check that the design is in the",
        "right ballpark for the 'multi-day operation' target claimed in the",
        "submission. It must be replaced with real current measurements on",
        "the chosen MCU/sensor parts during the 'Real-World Testing' phase.",
    ]

    report = "\n".join(lines)
    print(report)

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "power_latency_report.txt").write_text(report + "\n")
    print(f"\nReport saved to: {REPORTS_DIR / 'power_latency_report.txt'}")


if __name__ == "__main__":
    main()
