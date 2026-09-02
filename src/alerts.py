"""
Output Layer.

Simulates the two outputs named in the deck's architecture (Slide 6,
OUTPUT LAYER):

    - On-Wrist Buzzer + LED (instant, local)
    - BLE Alert -> Paired Phone / Caregiver App

No physical buzzer/LED/BLE radio exists yet in this prototype (see
README limitations), so both are simulated: the buzzer/LED as a console
representation of the beep/flash pattern, and the BLE alert as the exact
JSON payload that would be transmitted to a paired phone or caregiver app.
"""

import json
from datetime import datetime, timezone

from .config import STATE_NORMAL, STATE_COMBINED_EMERGENCY

# Beep pattern per state -- combined emergency escalates to a continuous
# tone, single-signal alerts use an intermittent pattern, matching the
# "escalated as the most urgent case" language from Slide 9.
BEEP_PATTERN = {
    STATE_COMBINED_EMERGENCY: "CONTINUOUS BEEP + FAST-FLASH LED",
}
DEFAULT_BEEP_PATTERN = "INTERMITTENT BEEP + FLASH LED"


def simulate_buzzer_led(state: str) -> str:
    """Return (and print) a text representation of the on-wrist alert."""
    if state == STATE_NORMAL:
        line = "[WRIST]  (silent - state Normal, no alert)"
    else:
        pattern = BEEP_PATTERN.get(state, DEFAULT_BEEP_PATTERN)
        line = f"[WRIST]  *** {state.upper()} *** -> {pattern}"
    print(line)
    return line


def simulate_ble_alert(state: str, fall_prob: float, heart_prob: float,
                        latency_ms: float) -> str:
    """
    Build and print the JSON alert payload that would be pushed over BLE
    to a paired phone / caregiver app. Only non-Normal states are sent,
    matching the deck's privacy claim: "Only a short alert ever leaves
    the device -- never raw health data."
    """
    if state == STATE_NORMAL:
        return ""

    payload = {
        "device": "SentryBand",
        "state": state,
        "fall_probability": round(float(fall_prob), 3),
        "heart_probability": round(float(heart_prob), 3),
        "decision_latency_ms": round(float(latency_ms), 2),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Deliberately no raw accelerometer/PPG samples in this payload --
        # only the classification result, per the "raw health data never
        # leaves the wrist" design claim.
    }
    line = f"[BLE]    Alert sent to paired phone/caregiver app: {json.dumps(payload)}"
    print(line)
    return line
