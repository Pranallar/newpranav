"""
EEG Balloon Game — One-command launcher.
Runs the HTTP server, WebSocket server, and EEG simulation all together.

Usage:  python run_balloon.py
"""
import asyncio
import json
import logging
import subprocess
import sys
import threading
import webbrowser
import http.server
import socketserver
from pathlib import Path

import numpy as np
import joblib
import websockets
from scipy.signal import welch, butter, sosfilt, iirnotch, lfilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("balloon")

BANDS = {"delta":(1,4), "theta":(4,8), "alpha":(8,13), "beta":(13,30), "gamma":(30,40)}
SCALP = ["F3","F4","C3","Cz","C4","P3","P4"]
MODEL_PATH = Path("model.pkl")
HTTP_PORT = 8080
WS_PORT = 5006
WS_CLIENTS = set()


def extract_features(window, sfreq, use_log=True):
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


# ── HTTP server (background thread) ─────────────────────────────────────
def start_http_server():
    """Serve balloon_game.html on HTTP_PORT in a background thread."""
    handler = http.server.SimpleHTTPRequestHandler
    handler.log_message = lambda *a: None  # Suppress logs
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", HTTP_PORT), handler) as httpd:
        httpd.serve_forever()


# ── WebSocket handler ────────────────────────────────────────────────────
async def ws_handler(websocket):
    global WS_CLIENTS
    WS_CLIENTS.add(websocket)
    log.info("🎮 Browser connected via WebSocket")
    try:
        async for _ in websocket:
            pass
    except Exception:
        pass
    finally:
        WS_CLIENTS.discard(websocket)
        log.info("🎮 Browser disconnected")


async def send_to_game(data):
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


# ── Simulation ───────────────────────────────────────────────────────────
async def run_simulation():
    import pyxdf, mne
    mne.set_log_level("ERROR")

    bundle = joblib.load(MODEL_PATH)
    if isinstance(bundle, dict):
        model, use_log = bundle["pipeline"], bundle.get("use_log", False)
    else:
        model, use_log = bundle, False
    log.info(f"📦 Model loaded (log={use_log})")

    XDF = "sub-P001_ses-S001_task-Default_run-001_eeg (1).xdf"
    streams, _ = pyxdf.load_xdf(XDF)
    eeg_s = [s for s in streams if s["info"]["type"][0] == "EEG"][0]
    mk_s = [s for s in streams if s["info"]["type"][0] == "Markers"
             and len(s.get("time_series", [])) > 0][0]

    data = eeg_s["time_series"].T
    sfreq = float(eeg_s["info"]["nominal_srate"][0])
    ch_labels = [c["label"][0] for c in eeg_s["info"]["desc"][0]["channels"][0]["channel"]]
    idx = [ch_labels.index(c) for c in SCALP]
    eeg = data[idx] * 1e-6

    sos = butter(4, [1.0, 40.0], btype="band", fs=sfreq, output="sos")
    for i in range(eeg.shape[0]): eeg[i] = sosfilt(sos, eeg[i])
    b60, a60 = iirnotch(60.0, 30.0, sfreq)
    for i in range(eeg.shape[0]): eeg[i] = lfilter(b60, a60, eeg[i])
    eeg -= eeg.mean(axis=0, keepdims=True)

    t0 = eeg_s["time_stamps"][0]
    trials = []
    for m, t in zip(mk_s["time_series"], mk_s["time_stamps"]):
        if m[0].endswith("_start"):
            truth = "concentration" if "focus" in m[0] else "relaxation"
            trials.append((t - t0, truth))

    n_w = int(3.0 * sfreq)
    n_s = int(1.0 * sfreq)

    log.info("⏳ Waiting for browser to connect…")
    while not WS_CLIENTS:
        await asyncio.sleep(0.3)
    log.info("✅ Connected! Starting in 2 seconds…\n")
    await asyncio.sleep(2)

    correct, total = 0, 0
    log.info(f"▶ Replaying {len(trials)} trials\n")

    for start_sec, truth in trials:
        start = int(start_sec * sfreq)
        for off in range(0, int(10 * sfreq) - n_w + 1, n_s):
            w0, w1 = start + off, start + off + n_w
            if w1 > eeg.shape[1]: break

            window = eeg[:, w0:w1]
            features = extract_features(window, sfreq, use_log=use_log)
            pred = model.predict(features)[0]
            pred_name = "relaxation" if pred == 1 else "concentration"

            total += 1
            if pred_name == truth: correct += 1
            acc_str = f"{correct/total:.0%}"

            symbol = "🔴" if pred_name == "concentration" else "🟢"
            match = "✓" if pred_name == truth else "✗"
            log.info(f"  {symbol} {pred_name:15s} (truth={truth:15s}) {match}  acc={acc_str}")

            await send_to_game({
                "state": pred_name, "truth": truth,
                "n": total, "acc": acc_str
            })
            await asyncio.sleep(1.0)

    log.info(f"\n{'='*50}")
    log.info(f"  Done: {correct}/{total} ({correct/total:.1%})")
    log.info(f"{'='*50}")
    while True:
        await asyncio.sleep(10)


# ── Main ─────────────────────────────────────────────────────────────────
async def main():
    # Start HTTP in background thread
    t = threading.Thread(target=start_http_server, daemon=True)
    t.start()
    log.info(f"🌐 HTTP server on http://localhost:{HTTP_PORT}")

    # Start WebSocket server
    ws_server = await websockets.serve(ws_handler, "127.0.0.1", WS_PORT)
    log.info(f"🔌 WebSocket server on ws://127.0.0.1:{WS_PORT}")

    # Update balloon_game.html to auto-connect (use the right URL)
    log.info(f"\n🎈 Opening balloon game…\n")
    webbrowser.open(f"http://localhost:{HTTP_PORT}/balloon_game.html")

    await run_simulation()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("\n👋 Stopped.")
