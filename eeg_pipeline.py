"""
EEG State Classification: Focus vs. Relaxation
================================================
Complete pipeline implementing the full notebook:
  1. Load XDF recording → MNE Raw object
  2. Preprocess: bandpass, notch, re-reference
  3. Epoch with artifact rejection
  4. Analysis A — Alpha-band power comparison (Focus vs. Relaxation)
  5. Analysis B — LDA & SVM classification with spectral features

Usage:
    python eeg_pipeline.py --xdf path/to/recording.xdf
    python eeg_pipeline.py --xdf sub-P001_ses-S001_task-Default_run-001_eeg.xdf
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless-safe backend; swap to "TkAgg" for interactive plots
import matplotlib.pyplot as plt
import numpy as np
import pyxdf
import mne

from load_session import load_session
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SCALP_CHANNELS = ["F3", "F4", "C3", "Cz", "C4", "P3", "P4"]

ALPHA_LOW, ALPHA_HIGH = 8.0, 13.0

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}

OUTPUT_DIR = Path("results")


# ─────────────────────────────────────────────
# Step 1 — Load XDF Recording
# ─────────────────────────────────────────────
# All loading logic lives in load_session.py.  The functions below are thin
# compatibility shims so older call-sites keep working.

def load_xdf_and_build_raw(xdf_path: str) -> tuple:
    """Load XDF file → MNE Raw via the canonical ``load_session`` function.

    Returns ``(raw, ch_names, sfreq)`` to match the legacy call pattern.
    """
    print(f"\n{'='*60}")
    print("STEP 1 — Loading XDF Recording")
    print(f"{'='*60}")

    raw = load_session(xdf_path, scalp_channels=SCALP_CHANNELS, verbose=True)
    return raw, list(raw.ch_names), raw.info["sfreq"]


# ─────────────────────────────────────────────
# Step 2 — Preprocessing
# ─────────────────────────────────────────────

def preprocess(raw, ch_names):
    """Apply bandpass filter, notch filter, and average re-reference."""
    print(f"\n{'='*60}")
    print("STEP 2 — Preprocessing")
    print(f"{'='*60}")

    # 2.1 — Bandpass filter: keep 1–40 Hz
    print("Applying 1–40 Hz bandpass filter …")
    raw.filter(l_freq=1.0, h_freq=40.0, method="fir", fir_window="hamming")

    # 2.2 — Notch filter at 60 Hz (North American power line)
    print("Applying 60 Hz notch filter …")
    raw.notch_filter(freqs=60.0)

    # 2.3 — Average re-reference
    print("Setting average reference …")
    raw.set_eeg_reference("average", projection=False)

    # 2.4 — Verify via PSD plot
    fig = raw.compute_psd(fmax=80).plot(show=False)
    fig.suptitle("PSD after preprocessing", y=1.02)
    _save_fig(fig, "psd_after_preprocessing.png")
    print("Saved PSD plot → results/psd_after_preprocessing.png")

    return raw


# ─────────────────────────────────────────────
# Step 3 — Epoching
# ─────────────────────────────────────────────

def epoch(raw, sfreq, reject_threshold_uv=300):
    """Cut continuous data into labeled 10-second epochs."""
    print(f"\n{'='*60}")
    print("STEP 3 — Epoching")
    print(f"{'='*60}")

    events, event_id = mne.events_from_annotations(raw)

    tmin = 0.0
    tmax = 10.0 - 1.0 / sfreq   # exactly 10 s at this sample rate

    epochs = mne.Epochs(
        raw, events, event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        reject={"eeg": reject_threshold_uv * 1e-6},
        preload=True,
    )

    print(f"Event ID mapping: {event_id}")
    print(epochs)
    print("\nEpochs kept per condition:")
    counts = {cond: len(epochs[cond]) for cond in event_id}
    for cond, count in counts.items():
        print(f"  {cond}: {count}")

    # Sanity check — at least some epochs per condition
    empty_conditions = [cond for cond, count in counts.items() if count == 0]
    if empty_conditions:
        print(f"WARNING: No epochs for {empty_conditions}. "
              "Retrying with reject=None …")
        epochs = mne.Epochs(
            raw, events, event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            reject=None,
            preload=True,
        )
        counts = {cond: len(epochs[cond]) for cond in event_id}
        empty_conditions = [cond for cond, count in counts.items() if count == 0]
        if empty_conditions:
            raise RuntimeError(
                f"No epochs left for conditions: {empty_conditions}. "
                "Check that the marker names in the XDF end with '_start'."
            )

    return epochs, event_id


# ─────────────────────────────────────────────
# Step 4 — Analysis A: Alpha Power
# ─────────────────────────────────────────────

def compute_band_power(epochs_obj, fmin, fmax):
    """
    Compute mean power in a frequency band for each epoch.

    Returns
    -------
    np.ndarray of shape (n_epochs,) — one scalar per epoch,
    averaged across all channels.
    """
    if len(epochs_obj) == 0:
        raise RuntimeError("No epochs available for this condition.")

    # Welch PSD restricted to the band of interest
    psd = epochs_obj.compute_psd(method="welch", fmin=fmin, fmax=fmax)
    power = psd.get_data()           # (n_epochs, n_channels, n_freqs)
    return power.mean(axis=1).mean(axis=1)   # → (n_epochs,)


def analysis_alpha_power(epochs, event_id):
    """Compare alpha-band power between Focus and Relaxation trials."""
    print(f"\n{'='*60}")
    print("STEP 4 — Analysis A: Alpha Power")
    print(f"{'='*60}")

    # Resolve condition keys (handle both 'focus_start' and plain 'focus')
    focus_key = next((k for k in event_id if "focus" in k.lower()), None)
    relax_key = next((k for k in event_id if "relax" in k.lower()), None)

    if focus_key is None or relax_key is None:
        raise RuntimeError(
            f"Could not find focus/relax conditions in event_id: {event_id}"
        )

    alpha_focus = compute_band_power(epochs[focus_key], ALPHA_LOW, ALPHA_HIGH)
    alpha_relax = compute_band_power(epochs[relax_key], ALPHA_LOW, ALPHA_HIGH)

    print(f"Focus alpha power     : mean={alpha_focus.mean():.4e}, "
          f"std={alpha_focus.std():.4e}")
    print(f"Relaxation alpha power: mean={alpha_relax.mean():.4e}, "
          f"std={alpha_relax.std():.4e}")

    # ── Plot: box plot + histogram ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Alpha Power: Focus vs. Relaxation", fontsize=13, fontweight="bold")

    # Box plot
    axes[0].boxplot([alpha_focus, alpha_relax], labels=["Focus", "Relaxation"])
    axes[0].set_ylabel("Alpha power (V²/Hz)")
    axes[0].set_title("Alpha Power by Condition")

    for i, (data, color) in enumerate([(alpha_focus, "steelblue"),
                                        (alpha_relax, "salmon")]):
        x = np.random.default_rng(0).normal(i + 1, 0.04, size=len(data))
        axes[0].scatter(x, data, alpha=0.4, s=15, color=color)

    # Histogram
    axes[1].hist(alpha_focus, bins=15, alpha=0.5, label="Focus", color="steelblue")
    axes[1].hist(alpha_relax, bins=15, alpha=0.5, label="Relaxation", color="salmon")
    axes[1].set_xlabel("Alpha power (V²/Hz)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Alpha Power Distributions")
    axes[1].legend()

    plt.tight_layout()
    _save_fig(fig, "alpha_power_analysis.png")
    print("Saved alpha power plot → results/alpha_power_analysis.png")

    # ── Statistical test ────────────────────────────────────────────────────
    t_stat, p_value = stats.ttest_ind(alpha_focus, alpha_relax)
    print(f"\nTwo-sample t-test:  t = {t_stat:.3f},  p = {p_value:.4f}")
    if p_value < 0.05:
        print("→ Significant difference in alpha power (p < 0.05)")
    else:
        print("→ No significant difference detected (p ≥ 0.05)")

    return alpha_focus, alpha_relax


# ─────────────────────────────────────────────
# Step 5 — Analysis B: ML Classification
# ─────────────────────────────────────────────

def extract_band_features(epochs_obj):
    """
    Extract band-power features for each epoch.

    Returns
    -------
    X : np.ndarray of shape (n_epochs, n_channels * n_bands)
    feature_names : list[str]
    """
    all_band_powers = []
    feature_names = []

    for band_name, (fmin, fmax) in BANDS.items():
        psd = epochs_obj.compute_psd(fmin=fmin, fmax=fmax)
        power = psd.get_data()            # (n_epochs, n_channels, n_freqs)
        band_power = power.mean(axis=2)   # (n_epochs, n_channels) — avg over freqs
        all_band_powers.append(band_power)
        for ch in epochs_obj.ch_names:
            feature_names.append(f"{ch}_{band_name}")

    X = np.concatenate(all_band_powers, axis=1)   # (n_epochs, n_channels * n_bands)
    return X, feature_names


def analysis_classification(epochs, event_id):
    """Full ML pipeline: feature extraction → causal split → LDA + SVM → eval."""
    print(f"\n{'='*60}")
    print("STEP 5 — Analysis B: ML Classification")
    print(f"{'='*60}")

    # ── 5.1 Feature extraction ───────────────────────────────────────────────
    print("\n[5.1] Extracting band-power features …")
    X, feature_names = extract_band_features(epochs)
    print(f"Feature matrix: {X.shape}  ({X.shape[0]} epochs × {X.shape[1]} features)")
    print(f"Features: {feature_names}")

    # ── 5.2 Build labels & causal train/test split ───────────────────────────
    print("\n[5.2] Building labels and causal split …")
    id_to_name = {v: k for k, v in event_id.items()}
    labels = np.array([
        1 if "relax" in id_to_name[code] else 0
        for code in epochs.events[:, 2]
    ])
    n_total = len(labels)
    print(f"Labels (0=focus, 1=relaxation): {np.bincount(labels)}")

    split_idx = int(n_total * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    print(f"\nTrain: {len(y_train)} epochs  |  Test: {len(y_test)} epochs")
    print(f"Train balance: focus={np.sum(y_train==0)}, relaxation={np.sum(y_train==1)}")
    print(f"Test  balance: focus={np.sum(y_test==0)}, relaxation={np.sum(y_test==1)}")

    # ── 5.3 Train classifiers ────────────────────────────────────────────────
    print("\n[5.3] Training LDA and SVM …")

    lda_pipe = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    lda_pipe.fit(X_train, y_train)
    print("LDA trained ✓")

    svm_pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
    svm_pipe.fit(X_train, y_train)
    print("SVM (RBF) trained ✓")

    # ── 5.4 Evaluate ─────────────────────────────────────────────────────────
    print("\n[5.4] Evaluating on test set …")

    classifiers = {"LDA": lda_pipe, "SVM (RBF)": svm_pipe}
    fig, axes = plt.subplots(1, len(classifiers),
                             figsize=(5 * len(classifiers), 4))
    if len(classifiers) == 1:
        axes = [axes]

    for ax, (name, clf) in zip(axes, classifiers.items()):
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{'='*40}")
        print(f"{name}  —  Test accuracy: {acc:.1%}")
        print(f"{'='*40}")
        print(classification_report(y_test, y_pred,
                                    target_names=["focus", "relaxation"]))

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["focus", "relaxation"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}\nAccuracy: {acc:.1%}")

    plt.tight_layout()
    _save_fig(fig, "confusion_matrices.png")
    print("Saved confusion matrices → results/confusion_matrices.png")

    # ── 5.5 Feature importance (LDA weights) ─────────────────────────────────
    print("\n[5.5] LDA feature importance …")
    lda_model = lda_pipe.named_steps["lineardiscriminantanalysis"]
    coefs = lda_model.coef_.ravel()
    sorted_idx = np.argsort(np.abs(coefs))[::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(coefs)), coefs[sorted_idx], color="steelblue")
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=7)
    ax.set_xlabel("LDA coefficient")
    ax.set_title("Feature Importance (LDA weights)")
    ax.invert_yaxis()
    plt.tight_layout()
    _save_fig(fig, "lda_feature_importance.png")
    print("Saved feature importance plot → results/lda_feature_importance.png")

    print("\nTop 5 features by |weight|:")
    for rank, idx in enumerate(sorted_idx[:5], 1):
        print(f"  {rank}. {feature_names[idx]:20s}  weight = {coefs[idx]:+.4f}")

    return lda_pipe, svm_pipe, X_test, y_test


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _save_fig(fig, filename):
    """Save a matplotlib figure to the results directory."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EEG State Classification: Focus vs. Relaxation"
    )
    parser.add_argument(
        "--xdf",
        default="sub-P001_ses-S001_task-Default_run-001_eeg.xdf",
        help="Path to the XDF recording file",
    )
    parser.add_argument(
        "--reject-uv",
        type=float,
        default=300.0,
        help="Amplitude rejection threshold in µV (default: 300)",
    )
    args = parser.parse_args()

    xdf_path = Path(args.xdf)
    if not xdf_path.exists():
        raise FileNotFoundError(
            f"XDF file not found: {xdf_path}\n"
            f"Place the recording in the same directory and re-run, or pass --xdf <path>"
        )

    # ── Step 1 + 1.1: Load XDF → MNE Raw (canonical loader) ───────────────
    raw, ch_names, sfreq = load_xdf_and_build_raw(str(xdf_path))

    # ── Step 2: Preprocess ────────────────────────────────────────────────────
    raw = preprocess(raw, ch_names)

    # ── Step 3: Epoch ─────────────────────────────────────────────────────────
    epochs, event_id = epoch(raw, sfreq, reject_threshold_uv=args.reject_uv)

    # ── Step 4: Alpha power ───────────────────────────────────────────────────
    alpha_focus, alpha_relax = analysis_alpha_power(epochs, event_id)

    # ── Step 5: ML classification ─────────────────────────────────────────────
    lda_pipe, svm_pipe, X_test, y_test = analysis_classification(epochs, event_id)

    print(f"\n{'='*60}")
    print("Pipeline complete! Results saved to ./results/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
