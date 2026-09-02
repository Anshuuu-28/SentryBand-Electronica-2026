"""
End-to-end demo reproducing the deck's "How It Works -- Step by Step"
pipeline (Slide 5) on a simulated stream of sensor windows:

    1. Sensors Collect Data
    2. AI Checks the Pattern
    3. Decision Made On the Spot
    4. Alert Sent Right Away

Run from the project root (after scripts/train.py):
    python scripts/demo_realtime.py            # runs instantly
    python scripts/demo_realtime.py --live      # paces itself to real time
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
import numpy as np

from src.sensors import gen_accel_window, gen_ppg_window
from src.models import load_models
from src.pipeline import SentryBandPipeline
from src.alerts import simulate_buzzer_led, simulate_ble_alert
from src.config import STATE_NORMAL, WINDOW_SECONDS

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Weighted so most windows are Normal, with occasional emergencies --
# roughly what a real deployment would see.
SCENARIO_WEIGHTS = {
    "normal": 0.70,
    "fall": 0.12,
    "heart": 0.12,
    "combined": 0.06,
}


def pick_scenario(rng: np.random.Generator) -> str:
    names = list(SCENARIO_WEIGHTS.keys())
    probs = list(SCENARIO_WEIGHTS.values())
    return rng.choice(names, p=probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="number of windows to simulate")
    parser.add_argument("--live", action="store_true",
                         help="pace output to real time (sleep WINDOW_SECONDS between windows)")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    fall_clf, heart_clf = load_models(MODELS_DIR)
    pipeline = SentryBandPipeline(fall_clf, heart_clf)
    rng = np.random.default_rng(args.seed)

    print(f"SentryBand live-pipeline demo -- simulating {args.n} sensor windows")
    print(f"(each window = {WINDOW_SECONDS:.0f}s of accelerometer + PPG data)\n")

    for i in range(1, args.n + 1):
        scenario = pick_scenario(rng)
        is_fall = scenario in ("fall", "combined")
        is_heart = scenario in ("heart", "combined")

        # 1. Sensors Collect Data
        accel_win = gen_accel_window(is_fall, rng=rng)
        ppg_win = gen_ppg_window(is_heart, rng=rng)

        # 2 & 3. AI Checks the Pattern / Decision Made On the Spot
        result = pipeline.process_window(accel_win, ppg_win)

        print(f"--- Window {i:02d} (simulated ground truth: {scenario}) ---")
        print(f"[SENSE]  accel window std={np.std(accel_win):.3f}g, "
              f"ppg window std={np.std(ppg_win):.3f}")
        print(f"[AI]     fall_prob={result.fall_probability:.2f}  "
              f"heart_prob={result.heart_probability:.2f}  "
              f"(decision latency: {result.latency_ms:.2f} ms)")
        print(f"[DECIDE] state = {result.state}")

        # 4. Alert Sent Right Away
        simulate_buzzer_led(result.state)
        simulate_ble_alert(result.state, result.fall_probability,
                            result.heart_probability, result.latency_ms)
        print()

        if args.live and result.state == STATE_NORMAL:
            time.sleep(WINDOW_SECONDS)


if __name__ == "__main__":
    main()
