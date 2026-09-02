"""
Decision & Fusion Layer.

Reproduces the exact 4-state table from the submitted deck (Slide 9,
"Fusing Two Signals Into One Confident Call"):

    Normal              -> motion and heart rhythm both within expected range
    Possible Fall        -> sudden motion signature, heart rhythm unaffected
    Heart Alert          -> irregular/dangerous rhythm, motion stays normal
    Combined Emergency   -> fall AND abnormal heartbeat together (most urgent)
"""

from .config import (
    STATE_NORMAL, STATE_POSSIBLE_FALL, STATE_HEART_ALERT,
    STATE_COMBINED_EMERGENCY,
)

# Ordered from least to most urgent -- used by the alert system to decide
# escalation (e.g. continuous vs. intermittent buzzer pattern).
STATE_PRIORITY = {
    STATE_NORMAL: 0,
    STATE_POSSIBLE_FALL: 1,
    STATE_HEART_ALERT: 1,
    STATE_COMBINED_EMERGENCY: 2,
}


def fuse(fall_flag: bool, heart_flag: bool) -> str:
    """Combine the two independent binary inference outputs into one of
    the four named risk states."""
    if fall_flag and heart_flag:
        return STATE_COMBINED_EMERGENCY
    if fall_flag:
        return STATE_POSSIBLE_FALL
    if heart_flag:
        return STATE_HEART_ALERT
    return STATE_NORMAL
