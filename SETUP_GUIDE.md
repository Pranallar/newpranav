# 🧠 EEG Balloon Controller — Windows Setup Guide

## What You Need

- ✅ **Blackbird (X.on) headset** — charged, electrodes gelled
- ✅ **Windows laptop** with Python installed
- ✅ **This folder** copied onto the laptop (USB / Google Drive / GitHub)
- ✅ **X.on app** installed (for streaming EEG over Bluetooth)

---

## Step 1: Copy This Folder

Copy the entire `EEG Classification` folder to the Windows laptop.  
Put it somewhere easy like `C:\Users\YourName\Desktop\EEG Classification\`

---

## Step 2: Install Python (if not already installed)

Download from https://www.python.org/downloads/  
**IMPORTANT:** Check ✅ "Add Python to PATH" during installation.

Verify it works — open **Command Prompt** (search "cmd"):
```
python --version
```
Should show `Python 3.10+`

---

## Step 3: Install Dependencies

Open **Command Prompt**, navigate to the folder, install everything:

```
cd C:\Users\YourName\Desktop\EEG Classification
pip install mne pyxdf pylsl scikit-learn scipy numpy joblib matplotlib pyriemann
```

That's it. One command installs everything.

---

## Step 4: Connect the Blackbird Headset

1. **Turn on** the Blackbird headset
2. **Open the X.on app** on the laptop
3. **Pair via Bluetooth** — follow X.on's pairing instructions
4. In X.on, go to **Settings → LSL Streaming** and **enable it**
   - This makes the headset broadcast EEG data over your local network
   - You should see "LSL stream active" or similar
5. **Verify** the electrodes have good contact (check impedance in X.on)

> **What is LSL?** It's a protocol that lets apps share live data streams.  
> When X.on enables LSL, it creates a data stream called something like  
> `X.on-XXXXXX` that our Python script automatically finds and reads.

---

## Step 5: Test the Connection

With the headset streaming, run this in Command Prompt:

```
cd C:\Users\YourName\Desktop\EEG Classification
python -c "from pylsl import resolve_streams; streams = resolve_streams(5.0); print(f'Found {len(streams)} stream(s):'); [print(f'  {s.name()} - {s.type()} - {s.channel_count()} ch - {s.nominal_srate()} Hz') for s in streams]"
```

You should see something like:
```
Found 1 stream(s):
  X.on-102801-0065 - EEG - 11 ch - 250.0 Hz
```

**If you see 0 streams:** The headset isn't streaming. Go back to X.on and make sure LSL is enabled.

---

## Step 6: Run the Balloon Game

You need **2 things open at the same time:**

### Terminal 1: Start the classifier
```
cd C:\Users\YourName\Desktop\EEG Classification
python realtime_balloon.py
```

You'll see:
```
20:30:33 │ 🌐 WebSocket bridge on ws://127.0.0.1:5006
20:30:33 │ 🎈 Open balloon_game.html in your browser
20:30:33 │    Then click '🔌 Connect UDP' in the game
20:30:33 │ 🔍 Searching for EEG stream on LSL…
20:30:34 │ ✅ Found: X.on-102801-0065 (11 ch, 250.0 Hz)
20:30:34 │ 🧠 LIVE — predicting every 1.0s  (Ctrl+C to stop)
```

### Browser: Open the game
Double-click `balloon_game.html` (it opens in Chrome/Edge)  
Click the **🔌 Connect UDP** button in the top-right corner

### Play!
- **Concentrate** (do mental math: 1000 - 7 = 993 - 7 = 986...) → balloon inflates 🔴
- **Relax** (close eyes, breathe slowly, clear your mind) → balloon deflates 🟢
- The prediction updates every 1 second

---

## Quick Test WITHOUT the Headset

If you want to test everything works before connecting the headset:

### Option A: Simulation mode (replays recorded data)
```
python realtime_balloon.py --simulate
```
Then open `balloon_game.html` and click 🔌 Connect UDP

### Option B: Keyboard mode
Just open `balloon_game.html`, click **⌨ Keyboard**, then:
- Hold **C** = concentrate (inflate)
- Hold **R** = relax (deflate)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `python not found` | Reinstall Python, check "Add to PATH" |
| `No module named 'mne'` | Run `pip install mne pyxdf pylsl scikit-learn scipy numpy joblib matplotlib pyriemann` |
| `No EEG stream found` | Make sure X.on has LSL streaming enabled and headset is connected |
| `Cannot connect to WebSocket` | Make sure `python realtime_balloon.py` is running in a terminal FIRST |
| Balloon doesn't move | Click "🔌 Connect UDP" in the game. Make sure the terminal shows predictions |
| `model.pkl not found` | The model file must be in the same folder. It's included already |
| Predictions are all the same class | Electrode contact might be bad — check impedance in X.on |

---

## File Summary

| File | What it does |
|------|-------------|
| `balloon_game.html` | The visual balloon game (open in browser) |
| `realtime_balloon.py` | Reads EEG → classifies → sends to game |
| `model.pkl` | The trained brain-state classifier |
| `analyze_real_data.py` | Analyzes an XDF file and generates plots |
| `compare_models.py` | Compares different classifier approaches |
| `results/` | All the diagnostic plots |

---

## Recording New Data (to improve the model)

If you want to record more sessions to improve accuracy:

1. Install LabRecorder: https://github.com/labstreaminglayer/App-LabRecorder
2. Start the headset + X.on + LSL streaming
3. Run the experiment script: `python run_experiment.py`
4. In LabRecorder, check **BOTH**:
   - ☑ The EEG stream (X.on-XXXXX)
   - ☑ The marker stream (EEGStateMarkers)
5. Press Record, then press Enter in the experiment script
6. After recording, retrain: `python analyze_real_data.py`

Each new session improves accuracy. Current: **70%** with 1 session.  
Expected: **75-85%** with 3-5 sessions.
