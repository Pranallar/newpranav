"""
Real-time EEG classifier that reads from a Blackbird (X.on) headset
via LSL, classifies brain state, and sends commands to the balloon game.

Usage:
  Live mode:    python realtime_balloon.py
  Simulate:     python realtime_balloon.py --simulate
"""
import argparse
import asyncio
import json
import logging
import socket
import sys
import time
from pathlib import Path

import numpy as np
import joblib
from scipy.signal import welch, butter, sosfilt

# ── Config ───────────────────────────────────────────────────────────────
BANDS = {
    "delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
    "beta": (13, 30), "gamma": (30, 40),
}
SCALP = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]
WINDOW_SEC = 3.0
STRIDE_SEC = 1.0
SMOOTH_N = 3       # require N consecutive identical predictions before sending
MODEL_PATH = Path("model.pkl")
GAME_HOST = "127.0.0.1"
GAME_PORT = 5005    # UDP port for the game bridge
WS_PORT = 5006      # WebSocket port for the HTML balloon game

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("realtime")


# ── Feature extraction (matches training) ────────────────────────────────
def extract_features(window, sfreq, use_log=True):
    """Extract band-power features from a (n_channels, n_samples) array."""
    n_ch, n_samp = window.shape
    nperseg = min(n_samp, int(sfreq * 2))
    feats = []
    for ch in range(n_ch):
        f, pxx = welch(window[ch], fs=sfreq, nperseg=nperseg)
        for bname, (lo, hi) in BANDS.items():
            mask = (f >= lo) & (f < hi)
            val = pxx[mask].mean()
            feats.append(np.log(val + 1e-20) if use_log else val)
    return np.array(feats).reshape(1, -1)


# ── Bandpass filter ──────────────────────────────────────────────────────
def make_filter(sfreq, lo=1.0, hi=40.0, order=4):
    return butter(order, [lo, hi], btype="band", fs=sfreq, output="sos")


# ── WebSocket server (sends predictions to balloon_game.html) ────────────
class WebSocketBridge:
    def __init__(self, port=WS_PORT):
        self.port = port
        self.clients = set()

    async def handler(self, websocket):
        self.clients.add(websocket)
        log.info(f"🎮 Balloon game connected (WebSocket)")
        try:
            async for _ in websocket:
                pass  # We only send, not receive
        except Exception:
            pass
        finally:
            self.clients.discard(websocket)
            log.info(f"🎮 Balloon game disconnected")

    async def start(self):
        import websockets
        self.server = await websockets.serve(self.handler, "127.0.0.1", self.port)
        log.info(f"🌐 WebSocket bridge on ws://127.0.0.1:{self.port}")

    async def send(self, message):
        if not self.clients:
            return
        import websockets
        dead = set()
        for ws in self.clients.copy():
            try:
                await ws.send(message)
            except Exception:
                dead.add(ws)
        self.clients -= dead



# ── LSL reader ───────────────────────────────────────────────────────────
def find_eeg_stream():
    """Find the Blackbird/X.on EEG stream on LSL."""
    try:
        from pylsl import StreamInlet, resolve_streams
        log.info("🔍 Searching for EEG stream on LSL…")
        streams = resolve_streams(wait_time=5.0)
        eeg_streams = [s for s in streams if s.type().lower() == "eeg"]
        if not eeg_streams:
            log.error("❌ No EEG stream found. Is the headset streaming?")
            return None, None, None, None
        stream = eeg_streams[0]
        inlet = StreamInlet(stream, max_buflen=10)
        sfreq = stream.nominal_srate()
        n_ch = stream.channel_count()

        # Try to get channel names
        info = inlet.info()
        ch_names = []
        ch = info.desc().child("channels").child("channel")
        for _ in range(n_ch):
            ch_names.append(ch.child_value("label"))
            ch = ch.next_sibling()
        if not ch_names or ch_names[0] == "":
            ch_names = [f"Ch{i+1}" for i in range(n_ch)]

        log.info(f"✅ Found: {stream.name()} ({n_ch} ch, {sfreq} Hz)")
        log.info(f"   Channels: {ch_names}")
        return inlet, sfreq, n_ch, ch_names
    except ImportError:
        log.error("pylsl not installed. Run: pip3 install pylsl")
        return None, None, None, None


# ── Simulation mode ──────────────────────────────────────────────────────
async def run_simulation(ws_bridge):
    """Simulate brain states from the recorded XDF data."""
    log.info("🎭 SIMULATION MODE — replaying recorded session")

    bundle = joblib.load(MODEL_PATH)
    # Handle both old (bare pipeline) and new (dict) model formats
    if isinstance(bundle, dict):
        model = bundle["pipeline"]
        use_log = bundle.get("use_log", False)
    else:
        model = bundle
        use_log = False
    log.info(f"📦 Loaded model from {MODEL_PATH} (log={use_log})")

    # Load real data
    import pyxdf, mne
    from scipy.signal import iirnotch, lfilter
    mne.set_log_level("ERROR")
    XDF = "sub-P001_ses-S001_task-Default_run-001_eeg (1).xdf"
    streams, _ = pyxdf.load_xdf(XDF)
    eeg_stream = [s for s in streams if s["info"]["type"][0] == "EEG"][0]
    marker_stream = [s for s in streams if s["info"]["type"][0] == "Markers"
                     and len(s.get("time_series", [])) > 0][0]

    data = eeg_stream["time_series"].T
    sfreq = float(eeg_stream["info"]["nominal_srate"][0])
    ch_labels = [ch["label"][0] for ch in
                 eeg_stream["info"]["desc"][0]["channels"][0]["channel"]]
    scalp_idx = [ch_labels.index(ch) for ch in SCALP]
    eeg = data[scalp_idx] * 1e-6  # V

    # Filter (bandpass + notch + avg ref — matches training)
    sos = make_filter(sfreq)
    for i in range(eeg.shape[0]):
        eeg[i] = sosfilt(sos, eeg[i])
    b60, a60 = iirnotch(60.0, 30.0, sfreq)
    for i in range(eeg.shape[0]):
        eeg[i] = lfilter(b60, a60, eeg[i])
    eeg -= eeg.mean(axis=0, keepdims=True)  # avg reference

    # Get marker timing
    t0 = eeg_stream["time_stamps"][0]
    trial_info = []
    markers = marker_stream["time_series"]
    mtimes = marker_stream["time_stamps"]
    for m, t in zip(markers, mtimes):
        label = m[0]
        if label.endswith("_start"):
            truth = "concentration" if "focus" in label else "relaxation"
            trial_info.append((t - t0, truth))

    n_window = int(WINDOW_SEC * sfreq)
    n_stride = int(STRIDE_SEC * sfreq)
    history = []
    correct = 0
    total = 0

    # Wait for the game to connect before starting
    log.info("")
    log.info("⏳ Waiting for balloon game to connect…")
    log.info("   Open http://localhost:8080/balloon_game.html")
    log.info("   Then click '🔌 Connect UDP'")
    log.info("")
    while not ws_bridge.clients:
        await asyncio.sleep(0.5)
    log.info("🎮 Game connected! Starting in 3 seconds…\n")
    await asyncio.sleep(3)

    log.info(f"   Sliding {WINDOW_SEC}s windows every {STRIDE_SEC}s…")
    log.info(f"   {len(trial_info)} trials to replay\n")

    for start_sec, truth in trial_info:
        start_samp = int(start_sec * sfreq)
        for offset in range(0, int(10 * sfreq) - n_window + 1, n_stride):
            w0 = start_samp + offset
            w1 = w0 + n_window
            if w1 > eeg.shape[1]:
                break

            window = eeg[:, w0:w1]
            features = extract_features(window, sfreq, use_log=use_log)
            pred = model.predict(features)[0]
            pred_name = "relaxation" if pred == 1 else "concentration"

            history.append(pred_name)
            if len(history) > SMOOTH_N:
                history = history[-SMOOTH_N:]

            # Smoothed prediction
            if len(set(history[-SMOOTH_N:])) == 1 and len(history) >= SMOOTH_N:
                smoothed = history[-1]
            else:
                smoothed = history[-1]

            total += 1
            if pred_name == truth:
                correct += 1

            # Send to game
            await ws_bridge.send(smoothed)

            # Log
            symbol = "🔴" if smoothed == "concentration" else "🟢"
            match = "✓" if pred_name == truth else "✗"
            time_in_trial = offset / sfreq
            log.info(f"  {symbol} {smoothed:15s} (truth={truth:15s}) {match}  "
                     f"t={start_sec + time_in_trial:.1f}s  "
                     f"acc={correct/total:.0%}")

            await asyncio.sleep(STRIDE_SEC)  # Real-time speed (1 pred/sec)

    log.info(f"\n{'='*50}")
    log.info(f"  Simulation done: {correct}/{total} correct ({correct/total:.1%})")
    log.info(f"{'='*50}")


# ── Live mode ────────────────────────────────────────────────────────────
async def run_live(ws_bridge):
    """Read from the Blackbird headset via LSL and classify in real-time."""
    inlet, sfreq, n_ch, ch_names = find_eeg_stream()
    if inlet is None:
        log.error("Cannot proceed without an EEG stream.")
        return

    model = joblib.load(MODEL_PATH)
    log.info(f"📦 Loaded model from {MODEL_PATH}")

    # Map headset channels to scalp channels
    scalp_idx = []
    for s in SCALP:
        if s in ch_names:
            scalp_idx.append(ch_names.index(s))
    if not scalp_idx:
        log.warning(f"⚠ None of {SCALP} found in {ch_names}. Using first {len(SCALP)} channels.")
        scalp_idx = list(range(min(len(SCALP), n_ch)))

    log.info(f"   Using channels: {[ch_names[i] for i in scalp_idx]}")

    n_window = int(WINDOW_SEC * sfreq)
    sos = make_filter(sfreq)
    buffer = np.zeros((n_ch, 0))
    history = []
    pred_count = 0

    log.info(f"\n🧠 LIVE — predicting every {STRIDE_SEC}s  (Ctrl+C to stop)\n")

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        # Pull samples
        samples, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=int(sfreq * STRIDE_SEC))
        if timestamps:
            chunk = np.array(samples).T * 1e-6  # (n_ch, n_new)
            buffer = np.hstack([buffer, chunk])

        # Keep only last WINDOW_SEC seconds
        if buffer.shape[1] > n_window * 2:
            buffer = buffer[:, -n_window * 2:]

        if buffer.shape[1] < n_window:
            await asyncio.sleep(0.05)
            continue

        # Extract window
        window = buffer[scalp_idx, -n_window:]

        # Filter
        filtered = np.zeros_like(window)
        for i in range(window.shape[0]):
            filtered[i] = sosfilt(sos, window[i])

        # Predict
        features = extract_features(filtered, sfreq)
        pred = model.predict(features)[0]
        pred_name = "relaxation" if pred == 1 else "concentration"

        history.append(pred_name)
        if len(history) > SMOOTH_N:
            history = history[-SMOOTH_N:]

        # Smoothed output
        if len(history) >= SMOOTH_N and len(set(history[-SMOOTH_N:])) == 1:
            smoothed = history[-1]
        else:
            smoothed = pred_name

        pred_count += 1
        symbol = "🔴" if smoothed == "concentration" else "🟢"
        log.info(f"  {symbol} {smoothed:15s}  (raw={pred_name})  #{pred_count}")

        # Send to balloon game
        await ws_bridge.send(smoothed)
        udp_sock.sendto(smoothed.encode(), (GAME_HOST, GAME_PORT))

        await asyncio.sleep(STRIDE_SEC)


# ── Main ─────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="EEG Balloon Controller - Real-time")
    parser.add_argument("--simulate", action="store_true",
                        help="Replay recorded data instead of live EEG")
    args = parser.parse_args()

    ws = WebSocketBridge(WS_PORT)
    await ws.start()

    log.info("")
    log.info("🎈 Open balloon_game.html in your browser")
    log.info("   Then click '🔌 Connect UDP' in the game")
    log.info("")

    if args.simulate:
        await run_simulation(ws)
    else:
        await run_live(ws)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("\n👋 Stopped.")
