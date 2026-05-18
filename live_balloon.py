"""
Live EEG Balloon Controller — Blackbird Headset Edition.

Connects to a Blackbird headset streaming via LSL (Lab Streaming Layer),
classifies brain activity in real-time, and controls the balloon game.

Setup:
  1. Turn on the Blackbird headset
  2. Open the Blackbird app and start the LSL stream
  3. Run: python live_balloon.py
  4. The balloon game opens in your browser automatically

Requirements:
  pip install pylsl websockets joblib numpy scipy scikit-learn
"""

import asyncio
import json
import logging
import threading
import webbrowser
import http.server
import socketserver
import time
import sys
from pathlib import Path
from collections import deque

import numpy as np
import joblib
import websockets
from scipy.signal import welch, butter, sosfilt, iirnotch, lfilter

# ── Configuration ────────────────────────────────────────────────────────
SCALP_CHANNELS = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 40)}
MODEL_PATH = Path("model.pkl")
HTTP_PORT = 8080
WS_PORT = 5006

# LSL stream settings — adjust these if your headset uses different names
LSL_STREAM_TYPE = "EEG"          # Most headsets use type "EEG"
LSL_STREAM_NAME = None           # None = connect to ANY EEG stream
                                 # Set to "Blackbird" or your stream name if needed

BUFFER_SECONDS = 3.0             # How many seconds of data to analyze at once
PREDICT_EVERY = 1.0              # How often to run a prediction (seconds)
SMOOTHING_WINDOW = 3             # Require N consecutive same-predictions to switch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("live")

WS_CLIENTS = set()


# ── Feature Extraction ──────────────────────────────────────────────────

def extract_features(window, sfreq):
    """Convert a (n_channels, n_samples) EEG window into a feature vector.

    Computes average band power in 5 frequency bands for each of 7 channels.
    Returns a (1, 35) array ready for model.predict().
    """
    n_ch, n_samp = window.shape
    nperseg = min(n_samp, int(sfreq * 2))
    feats = []
    for ch in range(n_ch):
        f, pxx = welch(window[ch], fs=sfreq, nperseg=nperseg)
        for bname, (lo, hi) in BANDS.items():
            mask = (f >= lo) & (f < hi)
            if mask.sum() == 0:
                feats.append(0.0)
            else:
                feats.append(pxx[mask].mean())
    return np.array(feats).reshape(1, -1)


# ── Preprocessing Filters ───────────────────────────────────────────────

def build_filters(sfreq):
    """Pre-compute filter coefficients (done once at startup)."""
    # Bandpass 1-40 Hz
    sos_bp = butter(4, [1.0, 40.0], btype="band", fs=sfreq, output="sos")
    # 60 Hz notch
    b_notch, a_notch = iirnotch(60.0, 30.0, sfreq)
    return sos_bp, b_notch, a_notch


def apply_filters(data, sos_bp, b_notch, a_notch):
    """Apply bandpass + notch filter to a (n_channels, n_samples) array."""
    filtered = np.empty_like(data)
    for i in range(data.shape[0]):
        temp = sosfilt(sos_bp, data[i])
        filtered[i] = lfilter(b_notch, a_notch, temp)
    # Average re-reference
    filtered -= filtered.mean(axis=0, keepdims=True)
    return filtered


# ── HTTP Server ──────────────────────────────────────────────────────────

def start_http_server():
    """Serve the balloon game HTML on a local port."""
    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *a: None
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", HTTP_PORT), handler) as httpd:
        httpd.serve_forever()


# ── WebSocket Handler ────────────────────────────────────────────────────

async def ws_handler(websocket):
    global WS_CLIENTS
    WS_CLIENTS.add(websocket)
    log.info("Browser connected!")
    try:
        async for _ in websocket:
            pass
    except Exception:
        pass
    finally:
        WS_CLIENTS.discard(websocket)
        log.info("Browser disconnected")


async def send_to_game(data):
    """Send a JSON message to all connected browsers."""
    global WS_CLIENTS
    if not WS_CLIENTS:
        return
    msg = json.dumps(data)
    dead = set()
    for ws in WS_CLIENTS.copy():
        try:
            await ws.send(msg)
        except Exception:
            dead.add(ws)
    for d in dead:
        WS_CLIENTS.discard(d)


# ── LSL Stream Discovery ────────────────────────────────────────────────

def find_eeg_stream():
    """Find and connect to the Blackbird headset's LSL stream.

    Returns (inlet, sfreq, channel_indices).
    """
    try:
        from pylsl import StreamInlet, resolve_streams
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"  ERROR importing pylsl: {e}")
        print("  Try: pip install pylsl")
        print("=" * 60)
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  Searching for Blackbird EEG stream...")
    print("  Make sure the headset is ON and the app is streaming")
    print("=" * 60 + "\n")

    # Try to find the stream
    log.info("Looking for EEG streams (waiting up to 30s)...")
    all_streams = resolve_streams(wait_time=10)

    # Filter to EEG streams (or by name if specified)
    if LSL_STREAM_NAME:
        streams = [s for s in all_streams if s.name() == LSL_STREAM_NAME]
    else:
        streams = [s for s in all_streams if s.type().upper() == "EEG"]

    if not streams and all_streams:
        log.info(f"No 'EEG' type found, using first available stream")
        streams = all_streams

    if not streams:
        print("\n" + "=" * 60)
        print("  No EEG stream found!")
        print()
        print("  Troubleshooting:")
        print("    1. Is the Blackbird headset turned on?")
        print("    2. Is the Blackbird app open and streaming?")
        print("    3. Are both devices on the same WiFi network?")
        print("    4. Try setting LSL_STREAM_NAME at the top of this file")
        print("       to match your headset's stream name.")
        print()
        print("  To see available streams, run:")
        print("    python -c \"from pylsl import resolve_streams; print(resolve_streams())\"")
        print("=" * 60)
        sys.exit(1)

    stream_info = streams[0]
    inlet = StreamInlet(stream_info, max_buflen=10)

    sfreq = stream_info.nominal_srate()
    n_channels = stream_info.channel_count()
    stream_name = stream_info.name()

    log.info(f"Found stream: '{stream_name}'")
    log.info(f"  Channels: {n_channels}")
    log.info(f"  Sample rate: {sfreq} Hz")

    # Try to get channel labels from the stream metadata
    ch_labels = []
    desc = stream_info.desc()
    channels_node = desc.child("channels")
    if not channels_node.empty():
        ch = channels_node.child("channel")
        while not ch.empty():
            label = ch.child_value("label")
            if label:
                ch_labels.append(label)
            ch = ch.next_sibling("channel")

    if len(ch_labels) == n_channels:
        log.info(f"  Channel labels: {ch_labels}")
    else:
        ch_labels = [f"CH{i+1}" for i in range(n_channels)]
        log.info(f"  No labels found, using generic: {ch_labels}")

    # Find which columns correspond to our 7 scalp channels
    channel_indices = []
    for scalp_ch in SCALP_CHANNELS:
        if scalp_ch in ch_labels:
            channel_indices.append(ch_labels.index(scalp_ch))
        else:
            log.warning(f"  Channel '{scalp_ch}' not found in stream!")

    if len(channel_indices) < len(SCALP_CHANNELS):
        log.warning(f"  Only found {len(channel_indices)}/{len(SCALP_CHANNELS)} channels")
        log.warning(f"  Using first {len(SCALP_CHANNELS)} channels as fallback")
        channel_indices = list(range(min(len(SCALP_CHANNELS), n_channels)))

    return inlet, sfreq, channel_indices


# ── Live EEG Loop ────────────────────────────────────────────────────────

async def run_live():
    """Main loop: read EEG from headset → classify → send to game."""

    # Load the trained model
    if not MODEL_PATH.exists():
        print(f"\nERROR: No trained model found at {MODEL_PATH}")
        print("Train one first: python train.py recording.xdf")
        sys.exit(1)

    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        model = bundle["pipeline"]
    else:
        model = bundle
    log.info(f"Model loaded from {MODEL_PATH}")

    # Connect to headset
    inlet, sfreq, ch_indices = find_eeg_stream()

    # Build filters
    sos_bp, b_notch, a_notch = build_filters(sfreq)
    n_buffer = int(BUFFER_SECONDS * sfreq)  # samples in 3-second window

    log.info(f"\nReady! Window: {BUFFER_SECONDS}s ({n_buffer} samples)")
    log.info(f"Predicting every {PREDICT_EVERY}s")

    # Wait for browser
    log.info("\nWaiting for browser to connect...")
    while not WS_CLIENTS:
        await asyncio.sleep(0.3)
    log.info("Browser connected! Starting predictions...\n")
    await asyncio.sleep(1)

    # Rolling buffer for EEG data
    n_ch = len(ch_indices)
    buffer = np.zeros((n_ch, n_buffer))  # (channels, samples)
    samples_collected = 0

    # Smoothing: require N consecutive same-predictions
    recent_preds = deque(maxlen=SMOOTHING_WINDOW)
    current_state = "neutral"
    total_preds = 0
    last_predict_time = time.time()

    while True:
        # Pull available samples from the headset
        samples, timestamps = inlet.pull_chunk(timeout=0.0)

        if samples:
            samples = np.array(samples)  # (n_pulled, n_all_channels)
            # Select only our scalp channels
            new_data = samples[:, ch_indices].T  # (n_ch, n_pulled)
            n_new = new_data.shape[1]

            # Shift buffer left and append new data
            if n_new >= n_buffer:
                buffer = new_data[:, -n_buffer:]
            else:
                buffer = np.roll(buffer, -n_new, axis=1)
                buffer[:, -n_new:] = new_data

            samples_collected += n_new

        # Predict every PREDICT_EVERY seconds
        now = time.time()
        if now - last_predict_time >= PREDICT_EVERY and samples_collected >= n_buffer:
            last_predict_time = now

            # Filter the buffer
            filtered = apply_filters(buffer.copy(), sos_bp, b_notch, a_notch)

            # Scale to Volts if needed (Blackbird typically sends µV)
            p99 = np.percentile(np.abs(filtered), 99)
            if p99 > 1e-3:
                filtered = filtered * 1e-6  # µV → V

            # Extract features and predict
            features = extract_features(filtered, sfreq)
            pred_label = model.predict(features)[0]
            pred_name = "relaxation" if pred_label == 1 else "concentration"

            # Smoothing: only switch state if N consecutive predictions agree
            recent_preds.append(pred_name)
            if len(recent_preds) == SMOOTHING_WINDOW and len(set(recent_preds)) == 1:
                current_state = recent_preds[0]

            total_preds += 1

            # Log and send
            symbol = "R" if current_state == "concentration" else "G"
            log.info(f"  [{symbol}] {current_state:15s}  (raw: {pred_name}, "
                     f"buffer: {list(recent_preds)}, n={total_preds})")

            await send_to_game({
                "state": current_state,
                "raw_pred": pred_name,
                "n": total_preds,
            })

        await asyncio.sleep(0.01)  # Don't spin the CPU


# ── Main Entry Point ─────────────────────────────────────────────────────

async def main():
    # Start HTTP server in background
    t = threading.Thread(target=start_http_server, daemon=True)
    t.start()
    log.info(f"HTTP server on http://localhost:{HTTP_PORT}")

    # Start WebSocket server
    ws_server = await websockets.serve(ws_handler, "127.0.0.1", WS_PORT)
    log.info(f"WebSocket server on ws://127.0.0.1:{WS_PORT}")

    # Open the balloon game
    log.info(f"Opening balloon game...")
    webbrowser.open(f"http://localhost:{HTTP_PORT}/balloon_game.html")

    # Start live EEG processing
    await run_live()


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  EEG Balloon Controller — LIVE HEADSET MODE")
    print("=" * 60)
    print()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("\nStopped by user.")
