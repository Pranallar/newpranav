"""
Real-time EEG classifier — Steps 15–17 of the project spec.

Reads live EEG from an LSL stream, maintains a rolling buffer,
applies the same preprocessing + feature extraction as offline,
and outputs smoothed predictions.

Usage:
    python realtime.py                  # uses model.pkl, prints predictions
    python realtime.py --model my.pkl   # custom model path
    python realtime.py --game           # also sends commands to balloon game
"""

from __future__ import annotations

import argparse
import json
import socket
import time
import logging
from collections import deque
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

from config import (
    SCALP_CHANNELS, SFREQ_EXPECTED,
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ,
    BUFFER_SECONDS, PREDICT_EVERY_S,
    SMOOTHING_MODE, MAJORITY_WINDOW, CONFIDENCE_THRESHOLD,
    GAME_HOST, GAME_PORT,
    MODEL_PATH, LOGS_DIR, CLASS_NAMES,
)
from features import extract_band_powers_from_array

logger = logging.getLogger(__name__)


class RealtimeClassifier:
    """Live EEG → prediction pipeline.

    Steps 15–17:
      15. LSL StreamInlet reads samples into a rolling buffer
      16. Same filter + features as offline (via sklearn Pipeline)
      17. Smoothing: majority vote or confidence threshold
    """

    def __init__(self, model_path: str = str(MODEL_PATH), send_to_game: bool = False):
        # Load the trained pipeline (filter → features → classifier)
        self.pipeline = joblib.load(model_path)
        logger.info("Loaded model from %s", model_path)

        self.sfreq = SFREQ_EXPECTED
        self.n_channels = len(SCALP_CHANNELS)
        self.buffer_size = int(BUFFER_SECONDS * self.sfreq)

        # Rolling buffer: (n_channels, buffer_size)
        self.buffer = np.zeros((self.n_channels, self.buffer_size))
        self.buffer_idx = 0  # how many samples have been pushed

        # Prediction smoothing
        self.prediction_history: deque = deque(maxlen=MAJORITY_WINDOW)

        # Game bridge
        self.send_to_game = send_to_game
        self._game_socket = None
        if send_to_game:
            self._game_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Prediction log
        LOGS_DIR.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = LOGS_DIR / f"realtime_log_{ts}.jsonl"
        self._log_file = open(self.log_path, "w")

    def push_samples(self, samples: np.ndarray) -> None:
        """Push new samples into the rolling buffer.

        Parameters
        ----------
        samples : (n_new_samples, n_channels)
        """
        samples = samples.T  # → (n_channels, n_new_samples)
        n_new = samples.shape[1]

        if n_new >= self.buffer_size:
            # More samples than buffer → just keep the last buffer_size
            self.buffer = samples[:, -self.buffer_size:]
        else:
            # Shift left and append
            self.buffer = np.roll(self.buffer, -n_new, axis=1)
            self.buffer[:, -n_new:] = samples

        self.buffer_idx += n_new

    def predict_current(self) -> dict:
        """Run prediction on the current buffer contents.

        Returns a dict with keys: class_id, class_name, confidence, smoothed_command
        """
        if self.buffer_idx < self.buffer_size:
            return {"class_id": -1, "class_name": "buffering", "confidence": 0.0,
                    "smoothed_command": None}

        # Step 16: Apply same feature extraction as offline
        features = extract_band_powers_from_array(self.buffer, self.sfreq)

        # Predict
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(features)[0]
            class_id = int(np.argmax(proba))
            confidence = float(proba[class_id])
        else:
            class_id = int(self.pipeline.predict(features)[0])
            confidence = 1.0

        class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")

        # Step 17: Smoothing
        self.prediction_history.append((class_id, confidence))
        smoothed = self._apply_smoothing()

        result = {
            "timestamp": datetime.now().isoformat(),
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 3),
            "smoothed_command": smoothed,
        }

        # Log every prediction (step 19)
        self._log_file.write(json.dumps(result) + "\n")
        self._log_file.flush()

        # Step 18: Send to game if enabled
        if self.send_to_game and smoothed is not None:
            self._send_game_command(smoothed)

        return result

    def _apply_smoothing(self) -> str | None:
        """Step 17: Don't fire commands on noisy single predictions."""
        if len(self.prediction_history) < MAJORITY_WINDOW:
            return None

        if SMOOTHING_MODE == "majority":
            recent_classes = [p[0] for p in self.prediction_history]
            if all(c == recent_classes[0] for c in recent_classes):
                return CLASS_NAMES.get(recent_classes[0], "unknown")
            return None

        elif SMOOTHING_MODE == "confidence":
            recent_confs = [p[1] for p in self.prediction_history]
            recent_classes = [p[0] for p in self.prediction_history]
            avg_conf = np.mean(recent_confs)
            if avg_conf >= CONFIDENCE_THRESHOLD and len(set(recent_classes)) == 1:
                return CLASS_NAMES.get(recent_classes[0], "unknown")
            return None

        return None

    def _send_game_command(self, command: str) -> None:
        """Step 18: Send classifier output to balloon game via UDP."""
        if self._game_socket is None:
            return
        try:
            msg = json.dumps({"command": command}).encode("utf-8")
            self._game_socket.sendto(msg, (GAME_HOST, GAME_PORT))
        except OSError as e:
            logger.warning("Failed to send game command: %s", e)

    def close(self):
        """Clean up resources."""
        self._log_file.close()
        if self._game_socket:
            self._game_socket.close()


def run_lsl_loop(model_path: str, send_to_game: bool = False):
    """Main real-time loop using pylsl.

    Step 15: StreamInlet pulls live samples.
    """
    try:
        from pylsl import StreamInlet, resolve_stream
    except ImportError:
        print("ERROR: pylsl is required for real-time mode.")
        print("Install with: pip install pylsl")
        return

    print("Looking for an EEG stream on LSL …")
    streams = resolve_stream("type", "EEG")
    if not streams:
        print("No EEG stream found. Make sure the headset is streaming.")
        return

    inlet = StreamInlet(streams[0])
    info = inlet.info()
    sfreq = info.nominal_srate()
    n_ch = info.channel_count()
    print(f"Connected: {info.name()} — {n_ch} channels @ {sfreq} Hz")

    classifier = RealtimeClassifier(model_path, send_to_game)
    classifier.sfreq = sfreq

    predict_interval = PREDICT_EVERY_S
    last_predict = time.time()

    print(f"\nRunning real-time classification (Ctrl+C to stop)")
    print(f"  Buffer: {BUFFER_SECONDS}s | Predict every: {predict_interval}s")
    print(f"  Smoothing: {SMOOTHING_MODE} (window={MAJORITY_WINDOW})")
    print(f"  Log: {classifier.log_path}\n")

    try:
        while True:
            # Pull all available samples
            chunk, timestamps = inlet.pull_chunk(timeout=0.0)
            if chunk:
                samples = np.array(chunk)
                # Select only scalp channels if more are present
                if samples.shape[1] > classifier.n_channels:
                    samples = samples[:, :classifier.n_channels]
                classifier.push_samples(samples)

            # Predict periodically
            now = time.time()
            if now - last_predict >= predict_interval:
                result = classifier.predict_current()
                last_predict = now

                if result["class_id"] >= 0:
                    cmd = result["smoothed_command"] or "---"
                    print(
                        f"  [{result['timestamp'][-12:]}]  "
                        f"{result['class_name']:15s}  "
                        f"conf={result['confidence']:.2f}  "
                        f"→ {cmd}"
                    )

            time.sleep(0.01)  # small sleep to prevent busy-waiting

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        classifier.close()


def main():
    parser = argparse.ArgumentParser(description="Real-time EEG Balloon Classifier")
    parser.add_argument(
        "--model", default=str(MODEL_PATH),
        help=f"Path to trained model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--game", action="store_true",
        help="Also send predictions to the balloon game via UDP",
    )
    args = parser.parse_args()
    run_lsl_loop(args.model, args.game)


if __name__ == "__main__":
    main()
