"""
Central configuration for the EEG Balloon Controller pipeline.

All tunable parameters live here so they can be changed in one place.
Both offline training and real-time prediction import from this file.
"""

from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Recording & channels
# ─────────────────────────────────────────────────────────────────────────────
SCALP_CHANNELS = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]
SFREQ_EXPECTED = 256.0  # nominal sample rate from the Blackbird headset

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
BANDPASS_LOW  = 1.0     # Hz  — removes slow drifts
BANDPASS_HIGH = 40.0    # Hz  — removes muscle noise, keeps alpha/beta/theta
NOTCH_FREQ    = 60.0    # Hz  — US power-line noise
REJECT_UV     = 100.0   # µV  — amplitude rejection threshold per channel

# ─────────────────────────────────────────────────────────────────────────────
# Epoching
# ─────────────────────────────────────────────────────────────────────────────
EPOCH_DURATION   = 3.0    # seconds per sliding window
EPOCH_STRIDE     = 1.0    # seconds between window starts (→ 2 s overlap)
TRIAL_DURATION   = 10.0   # seconds per labeled block in the protocol

# ─────────────────────────────────────────────────────────────────────────────
# Frequency bands for feature extraction
# ─────────────────────────────────────────────────────────────────────────────
BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 40),
}

# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────
CV_FOLDS         = 10      # StratifiedKFold splits
CAUSAL_SPLIT     = 0.8     # fraction of data used for training in causal split
MODEL_PATH       = Path("model.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# Real-time / LSL
# ─────────────────────────────────────────────────────────────────────────────
BUFFER_SECONDS   = 3.0     # rolling window size for live classification
PREDICT_EVERY_S  = 1.0     # how often to run prediction (seconds)

# ─────────────────────────────────────────────────────────────────────────────
# Smoothing (step 17 — prevent noisy single-prediction commands)
# ─────────────────────────────────────────────────────────────────────────────
SMOOTHING_MODE   = "majority"    # "majority" or "confidence"
MAJORITY_WINDOW  = 3             # last N predictions must agree
CONFIDENCE_THRESHOLD = 0.70      # probability threshold for "confidence" mode

# ─────────────────────────────────────────────────────────────────────────────
# Game bridge (step 18)
# ─────────────────────────────────────────────────────────────────────────────
GAME_HOST        = "127.0.0.1"
GAME_PORT        = 5555          # UDP port the balloon game listens on

# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────
RESULTS_DIR      = Path("results")
LOGS_DIR         = Path("logs")

# Class labels (used everywhere)
CLASS_NAMES      = {0: "concentration", 1: "relaxation"}
