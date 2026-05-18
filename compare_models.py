"""
Head-to-head comparison: Our model vs EEG-ML-Model-Neurotech.
Both run on the same XDF file, same epochs, same CV splits.
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.signal import welch

import mne; mne.set_log_level("ERROR")
import pyxdf, joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Import Model B (EEG-ML-Model) components
sys.path.insert(0, str(Path("EEG-ML-Model-Neurotech-main").resolve()))
from src.features import BandPowerFeatures
from src.pipelines import make_bandpower_pipeline, make_riemann_pipeline

RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
XDF = "sub-P001_ses-S001_task-Default_run-001_eeg (1).xdf"
SCALP = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]
BANDS = {"delta":(1,4), "theta":(4,8), "alpha":(8,13), "beta":(13,30), "gamma":(30,40)}

print("=" * 70)
print("  HEAD-TO-HEAD: Model A (Ours) vs Model B (EEG-ML-Model-Neurotech)")
print("=" * 70)

# ── Load & preprocess identically ────────────────────────────────────────
streams, _ = pyxdf.load_xdf(XDF)
eeg_stream = [s for s in streams if s["info"]["type"][0] == "EEG"][0]
marker_stream = [s for s in streams if s["info"]["type"][0] == "Markers"
                 and len(s.get("time_series", [])) > 0][0]

eeg_data = eeg_stream["time_series"].T
eeg_times = eeg_stream["time_stamps"]
sfreq = float(eeg_stream["info"]["nominal_srate"][0])
ch_labels = [ch["label"][0] for ch in eeg_stream["info"]["desc"][0]["channels"][0]["channel"]]
scalp_idx = [ch_labels.index(ch) for ch in SCALP]
eeg_scalp = eeg_data[scalp_idx] * 1e-6  # V

info = mne.create_info(ch_names=SCALP, sfreq=sfreq, ch_types="eeg")
raw = mne.io.RawArray(eeg_scalp, info, verbose=False)

markers = [m[0] for m in marker_stream["time_series"]]
mtimes = marker_stream["time_stamps"]
t0 = eeg_times[0]

# Inject annotations
onsets, descs = [], []
for t, m in zip(mtimes, markers):
    if m.endswith("_start"):
        mapped = "concentration" if "focus" in m else "relaxation"
        onsets.append(t - t0)
        descs.append(mapped)
raw.set_annotations(mne.Annotations(onset=onsets, duration=[0]*len(onsets), description=descs))

# Filter
raw.filter(1.0, 40.0, method="fir", verbose=False)
raw.notch_filter(60.0, verbose=False)

print(f"\n  Channels: {SCALP}")
print(f"  Duration: {raw.n_times/sfreq:.0f}s  |  Sfreq: {sfreq} Hz")
print(f"  Annotations: {len(onsets)} trials")

# ── Build shared epoch array (same for both models) ──────────────────────
class_labels = {"concentration": 0, "relaxation": 1}
n_window = int(3.0 * sfreq)
n_stride = int(1.0 * sfreq)
n_trial  = int(10.0 * sfreq)
data_full = raw.get_data()

X_3d, y_all = [], []
for ann in raw.annotations:
    desc = ann["description"]
    if desc not in class_labels: continue
    label = class_labels[desc]
    start = int(ann["onset"] * sfreq)
    for w0 in range(start, start + n_trial - n_window + 1, n_stride):
        w1 = w0 + n_window
        if w1 > raw.n_times: break
        window = data_full[:, w0:w1]
        # Reject at 150 µV (Model B default) — tightest common threshold
        if np.max(np.abs(window)) * 1e6 > 150:
            continue
        X_3d.append(window)
        y_all.append(label)

X_3d = np.stack(X_3d)
y = np.array(y_all)
n_conc = np.sum(y == 0); n_relax = np.sum(y == 1)
print(f"\n  Shared epochs: {len(y)} (conc={n_conc}, relax={n_relax})")
print(f"  Epoch shape: {X_3d.shape}  (epochs × channels × samples)")

# ── Build feature matrices ───────────────────────────────────────────────
# Model A features: manual band-power extraction (our approach)
def extract_bandpower_A(X_3d, sfreq):
    """Our model: raw Welch band power, no log, no scaling in features."""
    n_ep = X_3d.shape[0]
    feats, names = [], []
    for ep_idx in range(n_ep):
        row = []
        for ch_idx, ch in enumerate(SCALP):
            f, pxx = welch(X_3d[ep_idx, ch_idx], fs=sfreq, nperseg=min(n_window, int(sfreq*2)))
            for bname, (lo, hi) in BANDS.items():
                mask = (f >= lo) & (f < hi)
                row.append(pxx[mask].mean())
                if ep_idx == 0:
                    names.append(f"{ch}_{bname}")
        feats.append(row)
    return np.array(feats), names

X_A, fnames_A = extract_bandpower_A(X_3d, sfreq)

# ── Define all pipelines ─────────────────────────────────────────────────
pipelines = {
    # --- MODEL A variants (ours) ---
    "A: LDA (ours)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis()),
    ]),
    "A: SVM-RBF (ours)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale")),
    ]),
    "A: LogReg (ours)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
    ]),

    # --- MODEL B variants (EEG-ML-Model-Neurotech) ---
    "B: BandPower+LR": make_bandpower_pipeline(sfreq=sfreq),
    "B: Riemann+LR": make_riemann_pipeline(),
}

# ── Run 10-fold stratified CV ────────────────────────────────────────────
n_splits = min(10, n_conc, n_relax)
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

print(f"\n{'='*70}")
print(f"  {n_splits}-Fold Stratified Cross-Validation")
print(f"{'='*70}\n")

results = {}
for name, pipe in pipelines.items():
    # Model B pipelines expect 3D input (epochs × ch × samples)
    if name.startswith("B:"):
        X_in = X_3d
    else:
        X_in = X_A  # Model A uses pre-extracted features

    scores = cross_val_score(pipe, X_in, y, cv=kfold, scoring="accuracy")
    y_pred = cross_val_predict(pipe, X_in, y, cv=kfold)
    cm = confusion_matrix(y, y_pred)
    results[name] = {
        "mean": scores.mean(), "std": scores.std(),
        "scores": scores, "y_pred": y_pred, "cm": cm
    }
    print(f"  {name:25s}  →  {scores.mean():.1%} ± {scores.std():.1%}   "
          f"(best fold: {scores.max():.1%})")

# ── Per-class F1 for each ────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Detailed Classification Reports")
print(f"{'='*70}")
for name, r in results.items():
    print(f"\n─── {name} ───")
    print(classification_report(y, r["y_pred"],
          target_names=["concentration", "relaxation"], digits=3))

# ── Plot: accuracy comparison bar chart ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
names_sorted = sorted(results, key=lambda k: results[k]["mean"], reverse=True)
accs = [results[n]["mean"] for n in names_sorted]
stds = [results[n]["std"] for n in names_sorted]
colors = ["#6366f1" if n.startswith("A:") else "#f59e0b" for n in names_sorted]

bars = ax.barh(range(len(names_sorted)), accs, xerr=stds, height=0.5,
               color=colors, alpha=0.8, capsize=5, edgecolor="white", linewidth=1.5)
for i, (a, s) in enumerate(zip(accs, stds)):
    ax.text(a + s + 0.005, i, f"{a:.1%} ± {s:.1%}", va="center", fontsize=10, fontweight="bold")
ax.set_yticks(range(len(names_sorted)))
ax.set_yticklabels(names_sorted, fontsize=11)
ax.set_xlabel("Cross-Validation Accuracy", fontsize=12)
ax.set_xlim(0.4, 0.85)
ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")
ax.set_title("Model Comparison — Same Data, Same Splits", fontsize=14, fontweight="bold")
ax.legend(handles=[
    plt.Rectangle((0,0),1,1, fc="#6366f1", alpha=0.8, label="Model A (Ours)"),
    plt.Rectangle((0,0),1,1, fc="#f59e0b", alpha=0.8, label="Model B (EEG-ML-Model)"),
    plt.Line2D([0],[0], color="gray", linestyle="--", label="Chance (50%)"),
], fontsize=10)
ax.invert_yaxis()
ax.grid(True, axis="x", alpha=0.2)
plt.tight_layout()
fig.savefig(RESULTS / "comparison_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(); print(f"\n  Saved → {RESULTS / 'comparison_accuracy.png'}")

# ── Plot: confusion matrices side by side ────────────────────────────────
best_A = max([k for k in results if k.startswith("A:")], key=lambda k: results[k]["mean"])
best_B = max([k for k in results if k.startswith("B:")], key=lambda k: results[k]["mean"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Confusion Matrices (CV predictions)", fontsize=14, fontweight="bold")
for ax, name, color in [(axes[0], best_A, "Purples"), (axes[1], best_B, "YlOrBr")]:
    cm = results[name]["cm"]
    ConfusionMatrixDisplay(cm, display_labels=["Conc.", "Relax."]).plot(ax=ax, cmap=color, colorbar=False)
    ax.set_title(f"{name}\n{results[name]['mean']:.1%} ± {results[name]['std']:.1%}", fontweight="bold")
plt.tight_layout()
fig.savefig(RESULTS / "comparison_confusion.png", dpi=150, bbox_inches="tight")
plt.close(); print(f"  Saved → {RESULTS / 'comparison_confusion.png'}")

# ── Plot: per-fold accuracy curves ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
for name in [best_A, best_B]:
    marker = "o" if name.startswith("A:") else "s"
    ax.plot(range(1, n_splits+1), results[name]["scores"], marker=marker,
            linewidth=2, markersize=8, label=f"{name} ({results[name]['mean']:.1%})", alpha=0.8)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
ax.set_xlabel("Fold", fontsize=12); ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Per-Fold Accuracy", fontsize=14, fontweight="bold")
ax.set_ylim(0.35, 0.85); ax.legend(fontsize=11)
ax.grid(True, alpha=0.3); ax.set_xticks(range(1, n_splits+1))
plt.tight_layout()
fig.savefig(RESULTS / "comparison_folds.png", dpi=150, bbox_inches="tight")
plt.close(); print(f"  Saved → {RESULTS / 'comparison_folds.png'}")

# ── Summary ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  WINNER: {names_sorted[0]}  at  {results[names_sorted[0]]['mean']:.1%}")
print(f"{'='*70}")

winner_is_A = names_sorted[0].startswith("A:")
print(f"\n  Key differences between the models:")
print(f"    Model A (ours):  band power → StandardScaler → LDA/SVM")
print(f"                     Features: raw Welch PSD per band×channel")
print(f"    Model B (theirs): log band power → StandardScaler → LogReg")
print(f"                      Also has Riemannian: Covariances → TangentSpace → LR")
print(f"\n  Model B advantages:")
print(f"    • Uses log(power) — compresses dynamic range, often better for LR")
print(f"    • Riemannian pipeline captures cross-channel correlations")
print(f"    • class_weight='balanced' handles imbalanced data")
print(f"    • BandPowerFeatures is an sklearn transformer (cleaner for production)")
print(f"\n  Model A advantages:")
print(f"    • SVM-RBF handles non-linear boundaries (LR is linear only)")
print(f"    • LDA is specifically designed for 2-class discrimination")
