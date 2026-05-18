"""
Preprocessing module — Steps 6–8 of the project spec.

Handles filtering, artifact rejection, and sliding-window epoching.
"""

from __future__ import annotations

import numpy as np
import mne

from config import (
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ, REJECT_UV,
    EPOCH_DURATION, EPOCH_STRIDE, TRIAL_DURATION, BANDS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Filtering
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(raw: mne.io.RawArray, *, verbose: bool = True) -> mne.io.RawArray:
    """Bandpass + notch filter (in-place).

    - Bandpass 1–40 Hz: keeps neural oscillations, removes drift & muscle noise.
    - Notch 60 Hz: removes US power-line interference.
    - Average re-reference: standardizes the voltage baseline.
    """
    if verbose:
        print("  Bandpass 1–40 Hz …")
    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH,
               method="fir", fir_window="hamming", verbose=False)

    if verbose:
        print("  Notch 60 Hz …")
    raw.notch_filter(freqs=NOTCH_FREQ, verbose=False)

    if verbose:
        print("  Average re-reference …")
    raw.set_eeg_reference("average", projection=False, verbose=False)

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Artifact rejection
# ─────────────────────────────────────────────────────────────────────────────

def reject_dict(threshold_uv: float = REJECT_UV) -> dict:
    """Build the MNE reject dict: drop epochs where any channel > threshold."""
    return {"eeg": threshold_uv * 1e-6}


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Sliding-window epoching
# ─────────────────────────────────────────────────────────────────────────────

def epoch_sliding_window(
    raw: mne.io.RawArray,
    *,
    window_sec: float = EPOCH_DURATION,
    stride_sec: float = EPOCH_STRIDE,
    reject_uv: float = REJECT_UV,
    verbose: bool = True,
) -> tuple[mne.Epochs, dict]:
    """Slice the recording into overlapping labeled windows.

    For each *_start annotation, creates multiple 3-second windows
    (with 1-second stride) within the 10-second trial block.
    Windows that straddle a trial boundary are dropped.

    Returns (epochs, event_id).
    """
    sfreq = raw.info["sfreq"]

    # Get trial boundaries from annotations
    trials = []
    for ann in raw.annotations:
        label = ann["description"]
        onset = ann["onset"]
        if "focus" in label.lower() or "concentration" in label.lower():
            trials.append((onset, 0, label))
        elif "relax" in label.lower():
            trials.append((onset, 1, label))

    if not trials:
        raise RuntimeError("No focus/relaxation annotations found in the Raw object.")

    # Build sliding-window events
    events_list = []
    event_id = {}

    for trial_onset, class_id, label in trials:
        # Derive a clean event name
        event_name = "concentration" if class_id == 0 else "relaxation"
        if event_name not in event_id:
            event_id[event_name] = class_id + 1  # MNE event IDs are 1-indexed

        t = 0.0
        while t + window_sec <= TRIAL_DURATION:
            sample = int((trial_onset + t) * sfreq)
            if sample >= 0 and sample < raw.n_times:
                events_list.append([sample, 0, event_id[event_name]])
            t += stride_sec

    events = np.array(events_list, dtype=int)

    tmin = 0.0
    tmax = window_sec - 1.0 / sfreq

    epochs = mne.Epochs(
        raw, events, event_id,
        tmin=tmin, tmax=tmax,
        baseline=None,
        reject=reject_dict(reject_uv),
        preload=True,
        verbose=False,
    )

    if verbose:
        print(f"  Sliding window: {window_sec}s window, {stride_sec}s stride")
        print(f"  Events: {len(events)} windows → {len(epochs)} kept after rejection")
        for name, eid in event_id.items():
            n = len(epochs[name])
            print(f"    {name}: {n} epochs")

    return epochs, event_id
