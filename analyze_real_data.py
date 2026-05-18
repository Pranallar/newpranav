"""
Analyze the REAL EEG data from the participant's XDF file.
Generates visualizations and trains the classifier.
"""
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.signal import welch

import mne
import pyxdf
import joblib
mne.set_log_level("WARNING")

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import BANDS, CV_FOLDS, CLASS_NAMES
from features import extract_band_powers, make_lda_pipeline, make_svm_pipeline

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

XDF = "sub-P001_ses-S001_task-Default_run-001_eeg (1).xdf"
SCALP = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]

print("=" * 70)
print("  REAL EEG Data Analysis")
print("=" * 70)

# ── Load ─────────────────────────────────────────────────────────────────
streams, _ = pyxdf.load_xdf(XDF)
eeg_stream = [s for s in streams if s["info"]["type"][0] == "EEG"][0]
marker_stream = [s for s in streams if s["info"]["type"][0] == "Markers" and len(s.get("time_series", [])) > 0][0]

eeg_data = eeg_stream["time_series"].T  # (n_ch, n_samples)
eeg_times = eeg_stream["time_stamps"]
sfreq = float(eeg_stream["info"]["nominal_srate"][0])
ch_labels = [ch["label"][0] for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]]

# Select scalp channels only
scalp_idx = [ch_labels.index(ch) for ch in SCALP if ch in ch_labels]
scalp_names = [ch_labels[i] for i in scalp_idx]
eeg_scalp = eeg_data[scalp_idx]  # (7, n_samples)

# Scale µV → V
if np.abs(eeg_scalp).max() > 1e-3:
    eeg_scalp = eeg_scalp * 1e-6

print(f"\n  Channels: {scalp_names}")
print(f"  Sample rate: {sfreq} Hz")
print(f"  Duration: {eeg_scalp.shape[1]/sfreq:.1f}s ({eeg_scalp.shape[1]/sfreq/60:.1f} min)")
print(f"  Samples: {eeg_scalp.shape[1]:,}")

# ── Build MNE Raw ────────────────────────────────────────────────────────
info = mne.create_info(ch_names=scalp_names, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(eeg_scalp, info, verbose=False)

# Inject markers
markers = [m[0] for m in marker_stream["time_series"]]
mtimes = marker_stream["time_stamps"]
t0 = eeg_times[0]
onsets = [t - t0 for t, m in zip(mtimes, markers) if m.endswith("_start")]
descs = [m for m in markers if m.endswith("_start")]
raw.set_annotations(mne.Annotations(onset=onsets, duration=[0]*len(onsets), description=descs))
print(f"  Annotations: {len(onsets)} trial starts")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 1: Raw EEG overview
# ══════════════════════════════════════════════════════════════════════════
print("\n[1] Plotting raw EEG …")
data, times = raw[:, :]
fig, axes = plt.subplots(len(scalp_names), 1, figsize=(18, 2.2*len(scalp_names)), sharex=True)
fig.suptitle("Raw EEG — Full Recording (REAL DATA)", fontsize=14, fontweight="bold", y=1.01)
for i, (ch, ax) in enumerate(zip(scalp_names, axes)):
    sig = data[i] * 1e6
    ax.plot(times, sig, linewidth=0.15, color="#1e40af", alpha=0.8)
    ax.set_ylabel(f"{ch}\n(µV)", fontsize=8)
    p1, p99 = np.percentile(sig, [1, 99])
    ax.set_ylim(p1 - 10, p99 + 10)
    ax.grid(True, alpha=0.15)
    for ann in raw.annotations:
        t = ann["onset"]
        c = "#ef4444" if "focus" in ann["description"] else "#22c55e"
        ax.axvspan(t, t+10, alpha=0.06, color=c)
axes[-1].set_xlabel("Time (s)")
fig.legend(handles=[
    mpatches.Patch(facecolor="#ef4444", alpha=0.3, label="Concentration"),
    mpatches.Patch(facecolor="#22c55e", alpha=0.3, label="Relaxation"),
], loc="upper right", fontsize=10)
plt.tight_layout()
fig.savefig(RESULTS / "real_01_raw_eeg.png", dpi=150, bbox_inches="tight")
plt.close(); print(f"  → {RESULTS/'real_01_raw_eeg.png'}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 2: Zoomed 30s
# ══════════════════════════════════════════════════════════════════════════
print("[2] Zoomed view …")
n30 = int(60 * sfreq)
d30, t30 = raw[:, :n30]
fig, axes = plt.subplots(len(scalp_names), 1, figsize=(18, 2.2*len(scalp_names)), sharex=True)
fig.suptitle("Raw EEG — First 60 Seconds (Zoomed)", fontsize=14, fontweight="bold", y=1.01)
for i, (ch, ax) in enumerate(zip(scalp_names, axes)):
    sig = d30[i] * 1e6
    ax.plot(t30, sig, linewidth=0.4, color="#6d28d9")
    ax.set_ylabel(f"{ch}\n(µV)", fontsize=8)
    ax.grid(True, alpha=0.15)
    for ann in raw.annotations:
        t = ann["onset"]
        if t < 60:
            c = "#ef4444" if "focus" in ann["description"] else "#22c55e"
            ax.axvspan(t, min(t+10, 60), alpha=0.1, color=c)
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
fig.savefig(RESULTS / "real_02_zoomed.png", dpi=150, bbox_inches="tight")
plt.close(); print(f"  → {RESULTS/'real_02_zoomed.png'}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Filter
# ══════════════════════════════════════════════════════════════════════════
print("[3] Filtering …")
raw.filter(1.0, 40.0, method="fir", fir_window="hamming", verbose=False)
raw.notch_filter(60.0, verbose=False)
raw.set_eeg_reference("average", projection=False, verbose=False)

fig = raw.compute_psd(fmax=50).plot(show=False)
fig.suptitle("PSD After Filtering (Real Data)", fontsize=12, fontweight="bold", y=1.02)
fig.savefig(RESULTS / "real_03_psd.png", dpi=150, bbox_inches="tight")
plt.close(); print(f"  → {RESULTS/'real_03_psd.png'}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: Epoch
# ══════════════════════════════════════════════════════════════════════════
print("[4] Epoching …")
events, event_id = mne.events_from_annotations(raw, verbose=False)
conc_key = next((k for k in event_id if "focus" in k.lower()), None)
relax_key = next((k for k in event_id if "relax" in k.lower()), None)

slide_events = []
for ann in raw.annotations:
    label = ann["description"]
    eid = event_id.get(label)
    if eid is None: continue
    t = 0.0
    while t + 3.0 <= 10.0:
        sample = int((ann["onset"] + t) * sfreq)
        if 0 <= sample < raw.n_times:
            slide_events.append([sample, 0, eid])
        t += 1.0
slide_events = np.array(slide_events, dtype=int)

epochs = mne.Epochs(raw, slide_events, event_id, tmin=0.0, tmax=3.0 - 1/sfreq,
                     baseline=None, reject={"eeg": 300e-6}, preload=True, verbose=False)
print(f"  {len(slide_events)} windows → {len(epochs)} kept after rejection")
for name in event_id:
    print(f"    {name}: {len(epochs[name])}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 4: PSD per class overlay
# ══════════════════════════════════════════════════════════════════════════
print("[5] PSD per class …")
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Power Spectrum: Concentration vs Relaxation (REAL DATA)", fontsize=13, fontweight="bold")
for key, label, color in [(conc_key, "Concentration", "#ef4444"), (relax_key, "Relaxation", "#22c55e")]:
    psd = epochs[key].compute_psd(fmax=45, verbose=False)
    psd_data = psd.get_data().mean(axis=0).mean(axis=0)
    ax.semilogy(psd.freqs, psd_data, color=color, linewidth=2, label=label, alpha=0.8)
ax.axvspan(8, 13, alpha=0.15, color="gold", label="Alpha band (8–13 Hz)")
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Power (V²/Hz)")
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(RESULTS / "real_04_psd_overlay.png", dpi=150, bbox_inches="tight")
plt.close(); print(f"  → {RESULTS/'real_04_psd_overlay.png'}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 5: Alpha power analysis
# ══════════════════════════════════════════════════════════════════════════
print("[6] Alpha power analysis …")
def get_alpha(ep):
    psd = ep.compute_psd(method="welch", fmin=8, fmax=13, verbose=False)
    return psd.get_data().mean(axis=1).mean(axis=1)

ac = get_alpha(epochs[conc_key])
ar = get_alpha(epochs[relax_key])
t_stat, p_val = stats.ttest_ind(ac, ar)
print(f"  Concentration alpha: {ac.mean():.4e} ± {ac.std():.4e}")
print(f"  Relaxation alpha:    {ar.mean():.4e} ± {ar.std():.4e}")
print(f"  t={t_stat:.3f}, p={p_val:.4e}")

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Alpha Power Analysis (8–13 Hz) — REAL DATA", fontsize=14, fontweight="bold")
bp = axes[0].boxplot([ac, ar], labels=["Concentration", "Relaxation"], patch_artist=True, widths=0.5)
bp["boxes"][0].set_facecolor("#ef444440"); bp["boxes"][1].set_facecolor("#22c55e40")
for j, (d, c) in enumerate([(ac, "#ef4444"), (ar, "#22c55e")]):
    x = np.random.default_rng(42).normal(j+1, 0.04, size=len(d))
    axes[0].scatter(x, d, alpha=0.4, s=12, color=c, zorder=3)
axes[0].set_ylabel("Alpha Power"); axes[0].set_title("Individual Epochs")
axes[1].hist(ac, bins=20, alpha=0.6, color="#ef4444", label="Concentration")
axes[1].hist(ar, bins=20, alpha=0.6, color="#22c55e", label="Relaxation")
axes[1].set_xlabel("Alpha Power"); axes[1].legend(); axes[1].set_title("Distribution")
sig = "***" if p_val<0.001 else "**" if p_val<0.01 else "*" if p_val<0.05 else "n.s."
axes[2].bar(["Conc.", "Relax."], [ac.mean(), ar.mean()], yerr=[ac.std(), ar.std()],
            color=["#ef4444","#22c55e"], alpha=0.7, capsize=8, width=0.5)
axes[2].set_ylabel("Mean Alpha Power"); axes[2].set_title(f"t={t_stat:.2f}, p={p_val:.2e} ({sig})")
plt.tight_layout()
fig.savefig(RESULTS / "real_05_alpha.png", dpi=150, bbox_inches="tight")
plt.close(); print(f"  → {RESULTS/'real_05_alpha.png'}")

# ══════════════════════════════════════════════════════════════════════════
# PLOT 6: Band power heatmap
# ══════════════════════════════════════════════════════════════════════════
print("[7] Band power heatmap …")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Band Power by Channel — REAL DATA", fontsize=13, fontweight="bold")
for ax, (key, title) in zip(axes, [(conc_key, "Concentration"), (relax_key, "Relaxation")]):
    matrix = []
    blabels = []
    for bn, (fmin, fmax) in BANDS.items():
        psd = epochs[key].compute_psd(fmin=fmin, fmax=fmax, verbose=False)
        matrix.append(psd.get_data().mean(axis=2).mean(axis=0))
        blabels.append(bn)
    matrix = np.log10(np.array(matrix) + 1e-20)
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(scalp_names))); ax.set_xticklabels(scalp_names)
    ax.set_yticks(range(len(blabels))); ax.set_yticklabels(blabels)
    ax.set_title(title, fontweight="bold"); plt.colorbar(im, ax=ax, label="log₁₀ Power")
plt.tight_layout()
fig.savefig(RESULTS / "real_06_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(); print(f"  → {RESULTS/'real_06_heatmap.png'}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: Train classifiers
# ══════════════════════════════════════════════════════════════════════════
print("\n[8] Training classifiers …")
X, fnames = extract_band_powers(epochs)
id2name = {v: k for k, v in event_id.items()}
y = np.array([1 if "relax" in id2name.get(c,"").lower() else 0 for c in epochs.events[:,2]])
print(f"  Features: {X.shape}  |  Labels: {np.sum(y==0)} conc, {np.sum(y==1)} relax")

pipes = {"LDA": make_lda_pipeline(), "SVM": make_svm_pipeline()}
n_splits = min(CV_FOLDS, min(np.sum(y==0), np.sum(y==1)))
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
res = {}
for name, pipe in pipes.items():
    scores = cross_val_score(pipe, X, y, cv=kfold, scoring="accuracy")
    res[name] = {"mean": scores.mean(), "std": scores.std()}
    print(f"    {name:10s}  {scores.mean():.1%} ± {scores.std():.1%}")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Classifier Performance — REAL DATA", fontsize=14, fontweight="bold")
for ax, (name, pipe) in zip(axes, pipes.items()):
    pipe.fit(X, y)
    cm = confusion_matrix(y, pipe.predict(X))
    ConfusionMatrixDisplay(cm, display_labels=["Concentration","Relaxation"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nCV: {res[name]['mean']:.1%} ± {res[name]['std']:.1%}", fontweight="bold")
plt.tight_layout()
fig.savefig(RESULTS / "real_07_confusion.png", dpi=150, bbox_inches="tight")
plt.close(); print(f"  → {RESULTS/'real_07_confusion.png'}")

# Save best model
best = max(res, key=lambda k: res[k]["mean"])
pipes[best].fit(X, y)
joblib.dump(pipes[best], "model.pkl")

print(f"\n{'='*70}")
print(f"  ✅ DONE — Best: {best} at {res[best]['mean']:.1%}")
print(f"  Model saved → model.pkl")
print(f"  Plots saved → results/real_*.png")
print(f"{'='*70}")
