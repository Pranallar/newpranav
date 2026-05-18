# 🎈 Neurotech — EEG Balloon Controller

**Real-time EEG classifier that inflates/deflates a balloon game based on two mental states: concentration vs. relaxation.**

Record brain activity → train a classifier → play a game with your mind.

---

## How It Works (Simple Version)

```
Your Brain → EEG Headset → Computer → "Are they concentrating?" → Balloon Goes Up/Down
```

### The Two Mental States

| State | What you do | What your brain does |
|-------|-------------|---------------------|
| **Concentration** | Silently subtract 7 from a number (1000, 993, 986…) | Alpha waves (8–13 Hz) get *quieter* — your brain is busy |
| **Relaxation** | Close eyes, breathe, do nothing | Alpha waves get *louder* — your brain is idling |

The classifier learns to tell these apart by looking at the power in different frequency bands of your EEG signal.

---

## Project Structure

```
EEG Classification/
├── config.py              # All tunable settings in one place
├── load_session.py        # XDF file → MNE Raw object (the universal loader)
├── preprocess.py          # Filtering + artifact rejection + epoching
├── features.py            # Band-power & Riemannian feature extraction
├── train.py               # Train classifier with 10-fold cross-validation
├── realtime.py            # Live EEG → predictions via LSL
├── game_bridge.py         # Sends predictions to the balloon game (UDP)
├── run_experiment.py      # Plays cues during data collection
├── eeg_pipeline.py        # Original all-in-one pipeline (still works)
├── eeg_classification *.ipynb  # Walkthrough notebook
├── requirements.txt       # Python dependencies
├── model.pkl              # Trained model (created by train.py)
├── results/               # Plots and figures
└── logs/                  # Real-time prediction logs
```

---

## Setup

```bash
# 1. Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## The Pipeline — Step by Step

The project follows a 20-step pipeline. Here's what each piece does:

### Phase 1: Data Collection (Steps 1–4)

#### Step 1 — Protocol
The experiment alternates between 10-second blocks:
```
[10s CONCENTRATE] → [4s rest] → [10s RELAX] → [4s rest] → repeat × 40
```
Total: ~27 minutes per person. Target: 40 trials per class per subject.

#### Step 2 — Recording
```bash
# Terminal 1: Start the cue script (pushes LSL markers)
python run_experiment.py --trials 40

# Terminal 2: LabRecorder records both EEG + markers into one .xdf file
```

**What's needed:**
- OpenBCI Blackbird headset (streaming via X.on app)
- LabRecorder (records all LSL streams into a single XDF file)
- The `run_experiment.py` script (plays cues, pushes markers)

#### Steps 3–4 — Pilot & Full Collection
Record 5–10 subjects. Track everything in a shared sheet.

### Phase 2: Offline Analysis (Steps 5–13)

#### Step 5 — Data Loading
```python
from load_session import load_session

raw = load_session("recording.xdf")
# Returns an MNE Raw object with trial markers as annotations
# All channel selection, scaling, and marker injection happens automatically
```

#### Step 6 — Filtering
```python
from preprocess import apply_filters

apply_filters(raw)
# Bandpass 1–40 Hz (keeps brain waves, removes noise)
# Notch 60 Hz (removes power-line interference)
# Average re-reference (standardizes voltage baseline)
```

#### Step 7 — Artifact Rejection
Epochs where any channel exceeds ±100 µV are automatically dropped.

#### Step 8 — Epoching
Instead of one 10-second chunk per trial, we use a **sliding window**:
- 3-second windows with 1-second stride (2s overlap)
- This creates ~8 windows per trial → more training data
- Windows that cross trial boundaries are dropped

#### Steps 9–13 — Train & Evaluate
```bash
# Train on one or more session files
python train.py recordings/session1.xdf recordings/session2.xdf

# This will:
#   - Load & preprocess each session
#   - Create sliding-window epochs
#   - Extract band-power features (5 bands × 7 channels = 35 features)
#   - Run 10-fold cross-validation with LDA and SVM
#   - Save the best model to model.pkl
#   - Print accuracy and gate check (need ≥65% to proceed)
```

#### Step 14 — Save Model
The trained sklearn Pipeline is saved as `model.pkl` via joblib. The real-time code loads this exact same pipeline so live and offline behavior match.

### Phase 3: Real-Time (Steps 15–19)

#### Step 15–16 — Live Classification
```bash
python realtime.py
# Connects to the live EEG stream via LSL
# Maintains a 3-second rolling buffer
# Every 1 second, extracts features and runs the model
```

#### Step 17 — Smoothing
Raw predictions are noisy. The smoother requires **3 consecutive identical predictions** before firing a command. This prevents the balloon from jittering.

#### Step 18 — Game Integration
```bash
# Terminal 1: Run the classifier (sends UDP commands)
python realtime.py --game

# Terminal 2: Run the game bridge (receives UDP, controls game)
python game_bridge.py

# Terminal 3: Run the balloon game
# (from the balloon_control_game repo)
```

Architecture:
```
[EEG Stream] → [realtime.py] --UDP:5555-→ [game_bridge.py] → [Balloon Game]
                    ↓
              [logs/realtime_log.jsonl]
```

The game never imports any ML code. Clean separation.

#### Step 19 — Closed-Loop Test
Every prediction is logged with a timestamp to `logs/realtime_log_*.jsonl` for post-session review.

### Phase 4: Demo (Step 20)

Record a video of it working. The README you're reading is the handoff doc.

---

## Key Concepts Explained Simply

### What is EEG?
Your brain cells communicate using tiny electrical signals. EEG (electroencephalography) measures these signals through electrodes on your scalp. The signals are **very small** (~50 microvolts — about 100,000× smaller than a AA battery).

### What are Brain Waves?
Brain signals oscillate at different speeds. We group them by frequency:

| Wave | Frequency | When it appears |
|------|-----------|----------------|
| **Delta** | 1–4 Hz | Deep sleep |
| **Theta** | 4–8 Hz | Drowsiness, memory |
| **Alpha** | 8–13 Hz | ⭐ **Relaxed, eyes closed** — our main signal |
| **Beta** | 13–30 Hz | Active thinking, focus |
| **Gamma** | 30–40 Hz | Complex processing |

### Why Alpha Waves?
When you close your eyes and relax, the back of your brain produces strong 8–13 Hz oscillations called **alpha waves**. When you concentrate (e.g., doing math), these waves get suppressed. This is one of the most reliable EEG signatures and exactly what our classifier detects.

### What is MNE?
MNE-Python is the standard open-source library for processing EEG data. Think of it as "pandas for brain data." It handles filtering, epoching, spectral analysis, and visualization.

### What is LSL?
Lab Streaming Layer (LSL) is a protocol for streaming time-series data (like EEG) over a network in real-time. The headset streams EEG data, our experiment script streams markers, and LabRecorder saves everything together.

### What is XDF?
The file format that LabRecorder saves. One `.xdf` file contains multiple synced streams (EEG + markers) with precise timestamps.

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `BANDPASS_LOW/HIGH` | 1 / 40 Hz | Filter range |
| `NOTCH_FREQ` | 60 Hz | Power-line noise frequency |
| `REJECT_UV` | 100 µV | Artifact threshold |
| `EPOCH_DURATION` | 3.0 s | Sliding window size |
| `EPOCH_STRIDE` | 1.0 s | Stride between windows |
| `CV_FOLDS` | 10 | Cross-validation folds |
| `SMOOTHING_MODE` | "majority" | How to smooth predictions |
| `MAJORITY_WINDOW` | 3 | Consecutive predictions needed |
| `GAME_PORT` | 5555 | UDP port for game bridge |

---

## Accuracy Expectations

| Level | Meaning |
|-------|---------|
| **≤ 50%** | Not working — check data quality |
| **50–65%** | Working but needs more data or tuning |
| **65%+** | ✅ Ready for real-time |
| **90%+** | Unrealistic on consumer EEG — likely overfitting |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No EEG stream found | Check X.on app is running, headset is connected |
| All epochs rejected | Lower `REJECT_UV` in config.py (try 200 or 300) |
| Accuracy ≤ 50% | Check electrode impedances (<50 kΩ), verify alpha contrast in PSD plots |
| Game not responding | Check both `realtime.py --game` and `game_bridge.py` are running |
| Alpha contrast invisible | Fix electrode placement — need good contact at P3, P4, Pz |

---

## Links

- [OpenBCI](https://openbci.com/)
- [MNE-Python](https://mne.tools/stable/)
- [PyRiemann](http://pyriemann.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [pylsl](https://github.com/labstreaminglayer/pylsl)
- [Balloon Game Repo](https://github.com/UChicago-Neurotech/balloon_control_game/)
- [LabRecorder](https://github.com/labstreaminglayer/App-LabRecorder)
