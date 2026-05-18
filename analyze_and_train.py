"""
Full demonstration of the pipeline using the real marker timing from
the participant's XDF file + simulated EEG data that mimics the expected
alpha-band contrast between concentration and relaxation.

When a real recording with both EEG + markers is available, replace the
simulated data path with the real XDF and the same pipeline runs unchanged.
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import welch

import mne
import pyxdf
import joblib
mne.set_log_level("WARNING")

from config import SCALP_CHANNELS, BANDS, CV_FOLDS, CLASS_NAMES
from features import extract_band_powers, make_lda_pipeline, make_svm_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Load the REAL markers from the participant's XDF
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  EEG Balloon Controller — Full Pipeline Demo")
print("  Using real marker timing from participant session")
print("=" * 70)

XDF_PATH = "sub-P001_ses-S001_task-Default_run-001_eeg.xdf"
streams, _ = pyxdf.load_xdf(XDF_PATH)
marker_stream = [s for s in streams if len(s.get("time_series", [])) > 0][0]
marker_strings = [m[0] for m in marker_stream["time_series"]]
marker_times = marker_stream["time_stamps"]

# Extract trial info
raw_start = marker_times[0]
trial_info = []
for i, (m, t) in enumerate(zip(marker_strings, marker_times)):
    if m.endswith("_start"):
        trial_info.append({"type": m, "onset": t - raw_start, "label": m})

n_focus = sum(1 for t in trial_info if "focus" in t["type"])
n_relax = sum(1 for t in trial_info if "relax" in t["type"])
session_duration = marker_times[-1] - marker_times[0] + 15  # add buffer

print(f"\n  Loaded {len(trial_info)} trial markers from {XDF_PATH}")
print(f"    Focus trials:       {n_focus}")
print(f"    Relaxation trials:  {n_relax}")
print(f"    Session duration:   {session_duration:.0f}s ({session_duration/60:.1f} min)")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: Generate simulated EEG data with realistic alpha contrast
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2] Generating simulated EEG data with alpha contrast …")
print("    (This models what the Blackbird headset WOULD produce)")

sfreq = 256.0  # Hz (Blackbird nominal rate)
n_channels = len(SCALP_CHANNELS)
n_samples = int(session_duration * sfreq)
rng = np.random.default_rng(42)

# Base noise (pink noise — more realistic than white)
def pink_noise(n_samples, rng):
    """Generate 1/f noise."""
    freqs = np.fft.rfftfreq(n_samples, d=1.0)
    freqs[0] = 1  # avoid divide by zero
    spectrum = 1.0 / np.sqrt(freqs)
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    return np.fft.irfft(spectrum * np.exp(1j * phases), n=n_samples)

# Generate base EEG for each channel
eeg_data = np.zeros((n_channels, n_samples))
for ch in range(n_channels):
    eeg_data[ch] = pink_noise(n_samples, rng) * 15  # ~15 µV baseline

# Add realistic alpha oscillations during relaxation trials
alpha_freq = 10.5  # Hz (center of alpha band)
t_array = np.arange(n_samples) / sfreq

for trial in trial_info:
    onset_sample = int(trial["onset"] * sfreq)
    end_sample = min(onset_sample + int(10 * sfreq), n_samples)

    if "relax" in trial["type"]:
        # RELAXATION: Strong alpha (8-13 Hz) — especially at P3, P4, Cz
        for ch_idx, ch_name in enumerate(SCALP_CHANNELS):
            alpha_amplitude = rng.uniform(8, 18)  # µV
            if ch_name in ("P3", "P4", "Cz"):
                alpha_amplitude *= 1.8  # stronger at parietal sites
            alpha = alpha_amplitude * np.sin(
                2 * np.pi * alpha_freq * t_array[onset_sample:end_sample]
                + rng.uniform(0, 2 * np.pi)
            )
            eeg_data[ch_idx, onset_sample:end_sample] += alpha

    elif "focus" in trial["type"]:
        # CONCENTRATION: Suppressed alpha + slight beta increase
        for ch_idx, ch_name in enumerate(SCALP_CHANNELS):
            # Weak residual alpha
            alpha_amplitude = rng.uniform(2, 5)  # much weaker
            alpha = alpha_amplitude * np.sin(
                2 * np.pi * alpha_freq * t_array[onset_sample:end_sample]
                + rng.uniform(0, 2 * np.pi)
            )
            eeg_data[ch_idx, onset_sample:end_sample] += alpha

            # Beta boost (18 Hz) during mental arithmetic
            beta = rng.uniform(3, 7) * np.sin(
                2 * np.pi * 18 * t_array[onset_sample:end_sample]
                + rng.uniform(0, 2 * np.pi)
            )
            eeg_data[ch_idx, onset_sample:end_sample] += beta

# Add some muscle artifact noise (occasional bursts)
for _ in range(20):
    burst_start = rng.integers(0, n_samples - int(0.5 * sfreq))
    burst_len = int(rng.uniform(0.1, 0.5) * sfreq)
    burst_ch = rng.integers(0, n_channels)
    eeg_data[burst_ch, burst_start:burst_start + burst_len] += rng.normal(0, 40, burst_len)

# Scale to Volts (MNE convention)
eeg_data_V = eeg_data * 1e-6

print(f"    Simulated: {n_channels} ch × {n_samples} samples ({session_duration:.0f}s @ {sfreq} Hz)")
print(f"    Alpha contrast: ~12 µV (relax) vs ~3 µV (focus)")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: Build MNE Raw object with real marker timing
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3] Building MNE Raw object …")

info = mne.create_info(ch_names=SCALP_CHANNELS, sfreq=sfreq,
                        ch_types=["eeg"] * n_channels)
raw = mne.io.RawArray(eeg_data_V, info, verbose=False)

# Inject the REAL markers from the participant's XDF
onsets = [t["onset"] for t in trial_info]
descriptions = [t["label"] for t in trial_info]
annotations = mne.Annotations(onset=onsets, duration=[0.0] * len(onsets),
                               description=descriptions)
raw.set_annotations(annotations)

print(f"    Raw: {raw.info['nchan']} ch, {raw.n_times} samples, {raw.times[-1]:.1f}s")
print(f"    Annotations: {len(raw.annotations)} trial markers")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: Plot raw EEG
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4] Plotting raw EEG …")

# Full recording overview
data, times = raw[:, :]
fig, axes = plt.subplots(n_channels, 1, figsize=(18, 2.2 * n_channels), sharex=True)
fig.suptitle("Simulated EEG — Full Recording\n(Using real participant trial timing)",
             fontsize=14, fontweight="bold", y=1.01)

for i, (ch, ax) in enumerate(zip(SCALP_CHANNELS, axes)):
    signal_uv = data[i] * 1e6
    ax.plot(times, signal_uv, linewidth=0.2, color="#1e40af", alpha=0.8)
    ax.set_ylabel(f"{ch}\n(µV)", fontsize=8)
    ax.set_ylim(-80, 80)
    ax.grid(True, alpha=0.15)

    for ann in raw.annotations:
        t = ann["onset"]
        if "focus" in ann["description"]:
            ax.axvspan(t, t + 10, alpha=0.08, color="#ef4444")
        elif "relax" in ann["description"]:
            ax.axvspan(t, t + 10, alpha=0.08, color="#22c55e")

axes[-1].set_xlabel("Time (s)")
from matplotlib.patches import Patch
fig.legend(handles=[
    Patch(facecolor="#ef4444", alpha=0.3, label="Concentration"),
    Patch(facecolor="#22c55e", alpha=0.3, label="Relaxation"),
], loc="upper right", fontsize=10)
plt.tight_layout()
fig.savefig(RESULTS / "01_raw_eeg.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {RESULTS / '01_raw_eeg.png'}")

# Zoomed view — first 60 seconds
fig, axes = plt.subplots(n_channels, 1, figsize=(18, 2.2 * n_channels), sharex=True)
fig.suptitle("EEG — First 60 Seconds (Zoomed)\nNote alpha oscillations during relaxation",
             fontsize=14, fontweight="bold", y=1.01)

data60, times60 = raw[:, :int(60 * sfreq)]
for i, (ch, ax) in enumerate(zip(SCALP_CHANNELS, axes)):
    signal_uv = data60[i] * 1e6
    ax.plot(times60, signal_uv, linewidth=0.5, color="#6d28d9")
    ax.set_ylabel(f"{ch}\n(µV)", fontsize=8)
    ax.set_ylim(-60, 60)
    ax.grid(True, alpha=0.15)
    for ann in raw.annotations:
        t = ann["onset"]
        if t < 60:
            if "focus" in ann["description"]:
                ax.axvspan(t, min(t + 10, 60), alpha=0.12, color="#ef4444")
            elif "relax" in ann["description"]:
                ax.axvspan(t, min(t + 10, 60), alpha=0.12, color="#22c55e")
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
fig.savefig(RESULTS / "02_eeg_zoomed_60s.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {RESULTS / '02_eeg_zoomed_60s.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Filter
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5] Filtering (1–40 Hz bandpass + 60 Hz notch) …")
raw.filter(1.0, 40.0, method="fir", fir_window="hamming", verbose=False)
raw.notch_filter(60.0, verbose=False)
raw.set_eeg_reference("average", projection=False, verbose=False)

# PSD after filtering
fig = raw.compute_psd(fmax=50).plot(show=False)
fig.suptitle("Power Spectral Density — After Filtering", fontsize=12, fontweight="bold", y=1.02)
fig.savefig(RESULTS / "03_psd_filtered.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {RESULTS / '03_psd_filtered.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: Epoch
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6] Epoching (3s sliding window, 1s stride) …")

events, event_id = mne.events_from_annotations(raw, verbose=False)

# Create sliding-window events manually
conc_key = next((k for k in event_id if "focus" in k.lower()), None)
relax_key = next((k for k in event_id if "relax" in k.lower()), None)

slide_events = []
for ann in raw.annotations:
    label = ann["description"]
    onset_s = ann["onset"]
    if "focus" in label.lower():
        eid = event_id[conc_key]
    elif "relax" in label.lower():
        eid = event_id[relax_key]
    else:
        continue

    t = 0.0
    while t + 3.0 <= 10.0:
        sample = int((onset_s + t) * sfreq)
        if 0 <= sample < raw.n_times:
            slide_events.append([sample, 0, eid])
        t += 1.0

slide_events = np.array(slide_events, dtype=int)
print(f"  Created {len(slide_events)} sliding windows")

epochs = mne.Epochs(raw, slide_events, event_id, tmin=0.0,
                     tmax=3.0 - 1.0 / sfreq, baseline=None,
                     reject={"eeg": 100e-6}, preload=True, verbose=False)

print(f"  Epochs after rejection: {len(epochs)}")
for name in event_id:
    n = len(epochs[name])
    print(f"    {name}: {n}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Alpha power analysis + visualization
# ═══════════════════════════════════════════════════════════════════════════
print("\n[7] Alpha power analysis …")

def get_alpha_power(epochs_obj):
    psd = epochs_obj.compute_psd(method="welch", fmin=8, fmax=13, verbose=False)
    return psd.get_data().mean(axis=1).mean(axis=1)

alpha_conc = get_alpha_power(epochs[conc_key])
alpha_relax = get_alpha_power(epochs[relax_key])

from scipy import stats
t_stat, p_val = stats.ttest_ind(alpha_conc, alpha_relax)

print(f"  Concentration alpha: {alpha_conc.mean():.4e} ± {alpha_conc.std():.4e}")
print(f"  Relaxation alpha:    {alpha_relax.mean():.4e} ± {alpha_relax.std():.4e}")
print(f"  t-test: t={t_stat:.3f}, p={p_val:.2e}")

# PSD overlay per class
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Power Spectrum: Concentration vs Relaxation\n(Alpha 8–13 Hz difference is the key signal)",
             fontsize=13, fontweight="bold")

for key, label, color in [(conc_key, "Concentration", "#ef4444"),
                           (relax_key, "Relaxation", "#22c55e")]:
    psd = epochs[key].compute_psd(fmax=45, verbose=False)
    psd_data = psd.get_data().mean(axis=0).mean(axis=0)  # avg epochs & channels
    freqs = psd.freqs
    ax.semilogy(freqs, psd_data, color=color, linewidth=2, label=label, alpha=0.8)

ax.axvspan(8, 13, alpha=0.15, color="gold", label="Alpha band (8–13 Hz)")
ax.set_xlabel("Frequency (Hz)", fontsize=11)
ax.set_ylabel("Power (V²/Hz)", fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(RESULTS / "04_psd_overlay.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {RESULTS / '04_psd_overlay.png'}")

# Alpha power comparison
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Alpha Power Analysis (8–13 Hz)\nConcentration vs Relaxation",
             fontsize=14, fontweight="bold")

# Box plot
bp = axes[0].boxplot([alpha_conc, alpha_relax],
                      labels=["Concentration", "Relaxation"], patch_artist=True,
                      widths=0.5)
bp["boxes"][0].set_facecolor("#ef444440")
bp["boxes"][1].set_facecolor("#22c55e40")
bp["boxes"][0].set_edgecolor("#ef4444")
bp["boxes"][1].set_edgecolor("#22c55e")
for i, (d, c) in enumerate([(alpha_conc, "#ef4444"), (alpha_relax, "#22c55e")]):
    x = np.random.default_rng(42).normal(i + 1, 0.04, size=len(d))
    axes[0].scatter(x, d, alpha=0.4, s=12, color=c, zorder=3)
axes[0].set_ylabel("Alpha Power (V²/Hz)")
axes[0].set_title("Individual Epochs")

# Histogram
axes[1].hist(alpha_conc, bins=20, alpha=0.6, color="#ef4444", label="Concentration")
axes[1].hist(alpha_relax, bins=20, alpha=0.6, color="#22c55e", label="Relaxation")
axes[1].set_xlabel("Alpha Power")
axes[1].set_ylabel("Count")
axes[1].set_title("Distribution")
axes[1].legend()

# Bar chart
means = [alpha_conc.mean(), alpha_relax.mean()]
sds = [alpha_conc.std(), alpha_relax.std()]
bars = axes[2].bar(["Conc.", "Relax."], means, yerr=sds,
                    color=["#ef4444", "#22c55e"], alpha=0.7, capsize=8, width=0.5)
axes[2].set_ylabel("Mean Alpha Power (V²/Hz)")
sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
axes[2].set_title(f"Mean ± SD\nt={t_stat:.2f}, p={p_val:.2e} ({sig})")

plt.tight_layout()
fig.savefig(RESULTS / "05_alpha_analysis.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {RESULTS / '05_alpha_analysis.png'}")

# Band power heatmap
print("\n  Band power heatmap …")
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Band Power by Channel\n(Relaxation should show more alpha at P3, P4)",
             fontsize=13, fontweight="bold")

for ax, (key, title) in zip(axes, [(conc_key, "Concentration"), (relax_key, "Relaxation")]):
    matrix = []
    band_labels = []
    for bname, (fmin, fmax) in BANDS.items():
        psd = epochs[key].compute_psd(fmin=fmin, fmax=fmax, verbose=False)
        power = psd.get_data().mean(axis=2).mean(axis=0)
        matrix.append(power)
        band_labels.append(bname)
    matrix = np.log10(np.array(matrix) + 1e-20)  # log scale for visibility
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(range(len(SCALP_CHANNELS)))
    ax.set_xticklabels(SCALP_CHANNELS, fontsize=10)
    ax.set_yticks(range(len(band_labels)))
    ax.set_yticklabels(band_labels, fontsize=10)
    ax.set_title(title, fontweight="bold", fontsize=12)
    plt.colorbar(im, ax=ax, label="log₁₀ Power")

plt.tight_layout()
fig.savefig(RESULTS / "06_band_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {RESULTS / '06_band_heatmap.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 8: Train classifiers
# ═══════════════════════════════════════════════════════════════════════════
print("\n[8] Training classifiers …")

X, feature_names = extract_band_powers(epochs)
id_to_name = {v: k for k, v in event_id.items()}
y = np.array([1 if "relax" in id_to_name.get(c, "").lower() else 0
              for c in epochs.events[:, 2]])

print(f"  Features: {X.shape} ({len(feature_names)} features per epoch)")
print(f"  Labels: {np.sum(y==0)} concentration, {np.sum(y==1)} relaxation")

pipelines = {"LDA": make_lda_pipeline(), "SVM (RBF)": make_svm_pipeline()}
n_splits = min(CV_FOLDS, min(np.sum(y == 0), np.sum(y == 1)))
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"\n  {n_splits}-fold Stratified Cross-Validation:")
results = {}
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X, y, cv=kfold, scoring="accuracy")
    results[name] = {"mean": scores.mean(), "std": scores.std(), "scores": scores}
    print(f"    {name:15s}  {scores.mean():.1%} ± {scores.std():.1%}")

best_name = max(results, key=lambda k: results[k]["mean"])
best_pipe = pipelines[best_name]
best_pipe.fit(X, y)

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Classifier Performance", fontsize=14, fontweight="bold")

for ax, (name, pipe) in zip(axes, pipelines.items()):
    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Concentration", "Relaxation"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name}\nCV: {results[name]['mean']:.1%} ± {results[name]['std']:.1%}",
                 fontweight="bold")

plt.tight_layout()
fig.savefig(RESULTS / "07_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved → {RESULTS / '07_confusion_matrices.png'}")

# Feature importance
if best_name == "LDA":
    lda = best_pipe.named_steps["lineardiscriminantanalysis"]
    coefs = lda.coef_.ravel()
    sorted_idx = np.argsort(np.abs(coefs))[::-1]

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#ef4444" if coefs[i] < 0 else "#22c55e" for i in sorted_idx]
    ax.barh(range(len(coefs)), coefs[sorted_idx], color=colors, alpha=0.75)
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=8)
    ax.set_xlabel("LDA Coefficient", fontsize=11)
    ax.set_title("Feature Importance\n← favors Concentration | favors Relaxation →",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.2)
    plt.tight_layout()
    fig.savefig(RESULTS / "08_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {RESULTS / '08_feature_importance.png'}")

    print(f"\n  Top 5 most important features:")
    for rank, idx in enumerate(sorted_idx[:5], 1):
        direction = "→ relax" if coefs[idx] > 0 else "→ conc"
        print(f"    {rank}. {feature_names[idx]:20s}  weight={coefs[idx]:+.4f}  {direction}")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 9: Save model
# ═══════════════════════════════════════════════════════════════════════════
model_path = Path("model.pkl")
joblib.dump(best_pipe, model_path)

acc = results[best_name]["mean"]
print(f"\n{'='*70}")
print(f"  ✅ PIPELINE COMPLETE")
print(f"{'='*70}")
print(f"  Best model: {best_name} — {acc:.1%} CV accuracy")
print(f"  Model saved: {model_path}")
print(f"  Plots saved: {RESULTS}/")
print(f"\n  Plots generated:")
for f in sorted(RESULTS.glob("*.png")):
    print(f"    📊 {f.name}")

if acc >= 0.65:
    print(f"\n  🎈 Ready for real-time balloon control!")
    print(f"     Run: python realtime.py --game")
