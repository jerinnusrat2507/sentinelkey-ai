"""
features.py
------------
Converts raw keyboard/mouse events collected over a time window into a
fixed-length numeric feature vector describing the user's behavioral
pattern (typing rhythm + mouse dynamics).

The exact same FEATURE_NAMES order must be used at both training time
and inference time, otherwise the model will score garbage.
"""

import math
from collections import defaultdict

# Order matters — this is the contract between training and inference.
FEATURE_NAMES = [
    "typing_speed_kps",      # key presses per second
    "avg_hold_time",         # avg (key release - key press) in seconds
    "avg_flight_time",       # avg (next key press - prev key release)
    "hold_time_std",         # std deviation of hold times (rhythm consistency)
    "flight_time_std",       # std deviation of flight times
    "avg_mouse_speed",       # avg pixels/second between consecutive moves
    "mouse_speed_std",       # std deviation of mouse speed
    "click_rate_cps",        # clicks per second
    "avg_click_interval",    # avg seconds between consecutive clicks
    "idle_ratio",            # fraction of the window with no input at all
]


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _std(values):
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def extract_features(key_events, mouse_move_events, mouse_click_events, window_seconds):
    """
    key_events: list of dicts {"key": str, "type": "press"|"release", "t": float}
    mouse_move_events: list of dicts {"x": float, "y": float, "t": float}
    mouse_click_events: list of dicts {"t": float}
    window_seconds: length of the capture window in seconds

    Returns: dict mapping FEATURE_NAMES -> float
    """
    window_seconds = max(window_seconds, 1e-6)

    # --- Keystroke dynamics ---
    press_times = {}
    hold_times = []
    release_events_ordered = []

    for ev in key_events:
        k = ev["key"]
        if ev["type"] == "press":
            # keep the *last* press time per key (handles OS key-repeat)
            press_times[k] = ev["t"]
        elif ev["type"] == "release" and k in press_times:
            hold = ev["t"] - press_times[k]
            if hold >= 0:
                hold_times.append(hold)
            release_events_ordered.append(ev["t"])
            del press_times[k]

    release_events_ordered.sort()
    flight_times = [
        release_events_ordered[i + 1] - release_events_ordered[i]
        for i in range(len(release_events_ordered) - 1)
        if release_events_ordered[i + 1] - release_events_ordered[i] < 3.0  # ignore long pauses
    ]

    num_key_presses = sum(1 for ev in key_events if ev["type"] == "press")
    typing_speed_kps = num_key_presses / window_seconds

    # --- Mouse dynamics ---
    mouse_move_events = sorted(mouse_move_events, key=lambda e: e["t"])
    speeds = []
    for i in range(1, len(mouse_move_events)):
        p0, p1 = mouse_move_events[i - 1], mouse_move_events[i]
        dt = p1["t"] - p0["t"]
        if dt <= 0:
            continue
        dist = math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])
        speeds.append(dist / dt)

    click_times = sorted(ev["t"] for ev in mouse_click_events)
    click_intervals = [
        click_times[i + 1] - click_times[i] for i in range(len(click_times) - 1)
    ]
    click_rate_cps = len(click_times) / window_seconds

    # --- Idle ratio: crude approximation based on largest gap between any events ---
    all_times = sorted(
        [ev["t"] for ev in key_events]
        + [ev["t"] for ev in mouse_move_events]
        + [ev["t"] for ev in mouse_click_events]
    )
    idle_time = 0.0
    for i in range(1, len(all_times)):
        gap = all_times[i] - all_times[i - 1]
        if gap > 1.0:  # gaps under 1s aren't "idle", just normal pacing
            idle_time += gap
    idle_ratio = min(idle_time / window_seconds, 1.0)

    return {
        "typing_speed_kps": typing_speed_kps,
        "avg_hold_time": _mean(hold_times),
        "avg_flight_time": _mean(flight_times),
        "hold_time_std": _std(hold_times),
        "flight_time_std": _std(flight_times),
        "avg_mouse_speed": _mean(speeds),
        "mouse_speed_std": _std(speeds),
        "click_rate_cps": click_rate_cps,
        "avg_click_interval": _mean(click_intervals),
        "idle_ratio": idle_ratio,
    }


def feature_dict_to_vector(feature_dict):
    """Ensures consistent ordering when handing features to the ML model."""
    return [feature_dict[name] for name in FEATURE_NAMES]
