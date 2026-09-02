"""
Configuration constants for the SentryBand prototype.

These values are taken directly from the "Design Targets" and
"System Architecture" sections of the submitted idea deck
(SentryBand.pptx, slides 6-7) and the idea-submission text answers.
Nothing here introduces a target that wasn't already stated in the
submission.
"""

# --- Sensing layer (Slide 6: SENSING LAYER) ---
SAMPLE_RATE_HZ = 25          # "Sampling @ 1-25 Hz continuous" -> upper bound used for prototype
WINDOW_SECONDS = 2.0         # "Raw Sensor Window (1-2 sec, accel + PPG)" -> upper bound used
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)

# --- Design targets (Slide 7: DESIGN TARGETS) ---
MODEL_FOOTPRINT_TARGET_KB = 50     # "< 50 KB quantized (int8)"
INFERENCE_LATENCY_TARGET_MS = 50   # "< 50 ms per decision window"

# --- Decision fusion (Slide 9: DECISION INTELLIGENCE) ---
# State names copied verbatim from the submitted deck / text answers.
STATE_NORMAL = "Normal"
STATE_POSSIBLE_FALL = "Possible Fall"
STATE_HEART_ALERT = "Heart Alert"
STATE_COMBINED_EMERGENCY = "Combined Emergency"

# --- Classifier thresholds ---
# Default operating point before the false-alarm tuning sweep
# (Slide 13: "False-Alarm Reduction: Tune thresholds and combined-signal
# logic to minimize false positives without missing true emergencies.")
DEFAULT_FALL_THRESHOLD = 0.60
DEFAULT_HEART_THRESHOLD = 0.70

RANDOM_SEED = 42
