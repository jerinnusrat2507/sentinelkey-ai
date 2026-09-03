"""
collector.py
------------
The desktop agent. Listens to keyboard + mouse activity in the background,
buckets it into fixed-size time windows, extracts a behavioral feature
vector per window, and either:

  1) POSTs it to the FastAPI backend for live trust scoring (default), or
  2) appends it to a local CSV for baseline data collection when a user
     runs with --record (used to build the training set for the model).

Usage:
    python collector.py                       # live mode -> sends to API
    python collector.py --record baseline.csv # data collection mode
    python collector.py --window 5            # change window size (seconds)
"""

import argparse
import csv
import os
import threading
import time

import requests
from pynput import keyboard, mouse

from features import FEATURE_NAMES, extract_features

API_URL = os.environ.get("SENTINELKEY_API_URL", "http://127.0.0.1:8000/ingest")


class ActivityBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.key_events = []
            self.mouse_move_events = []
            self.mouse_click_events = []
            self.window_start = time.time()

    def snapshot_and_reset(self):
        with self.lock:
            snapshot = (
                self.key_events,
                self.mouse_move_events,
                self.mouse_click_events,
                time.time() - self.window_start,
            )
            self.key_events = []
            self.mouse_move_events = []
            self.mouse_click_events = []
            self.window_start = time.time()
            return snapshot

    def add_key_event(self, key, event_type):
        with self.lock:
            self.key_events.append({"key": str(key), "type": event_type, "t": time.time()})

    def add_mouse_move(self, x, y):
        with self.lock:
            self.mouse_move_events.append({"x": x, "y": y, "t": time.time()})

    def add_mouse_click(self):
        with self.lock:
            self.mouse_click_events.append({"t": time.time()})


def build_listeners(buf: ActivityBuffer):
    def on_press(key):
        buf.add_key_event(key, "press")

    def on_release(key):
        buf.add_key_event(key, "release")

    def on_move(x, y):
        buf.add_mouse_move(x, y)

    def on_click(x, y, button, pressed):
        if pressed:
            buf.add_mouse_click()

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)
    return kb_listener, mouse_listener


def ensure_csv_header(path):
    file_exists = os.path.isfile(path)
    if not file_exists:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(FEATURE_NAMES)


def run(window_seconds: float, record_path: str | None):
    buf = ActivityBuffer()
    kb_listener, mouse_listener = build_listeners(buf)
    kb_listener.start()
    mouse_listener.start()

    if record_path:
        ensure_csv_header(record_path)

    print(f"[SentinelKey] Agent running. Window={window_seconds}s "
          f"Mode={'RECORD -> ' + record_path if record_path else 'LIVE -> ' + API_URL}")
    print("[SentinelKey] Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(window_seconds)
            key_events, move_events, click_events, elapsed = buf.snapshot_and_reset()
            features = extract_features(key_events, move_events, click_events, elapsed)

            if record_path:
                with open(record_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([features[name] for name in FEATURE_NAMES])
                print(f"[recorded] {features}")
            else:
                try:
                    resp = requests.post(API_URL, json=features, timeout=3)
                    trust_score = resp.json().get("trust_score", "?")
                    print(f"[sent] trust_score={trust_score}  features={features}")
                except requests.RequestException as e:
                    print(f"[warn] could not reach backend: {e}")
    except KeyboardInterrupt:
        print("\n[SentinelKey] Stopping agent...")
    finally:
        kb_listener.stop()
        mouse_listener.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelKey AI behavioral data collector")
    parser.add_argument("--window", type=float, default=5.0, help="Window size in seconds")
    parser.add_argument("--record", type=str, default=None,
                         help="If set, save feature vectors to this CSV instead of calling the API")
    args = parser.parse_args()
    run(args.window, args.record)
