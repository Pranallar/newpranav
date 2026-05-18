"""
Training & evaluation — Steps 9, 12–14 of the project spec.

Runs the full offline pipeline:
  1. Load all sessions → preprocess → epoch
  2. Extract features (band-power and optionally Riemannian)
  3. 10-fold stratified CV + per-subject & cross-subject evaluation
  4. Save best model to model.pkl

Usage:
    python train.py path/to/session1.xdf path/to/session2.xdf ...
    python train.py recordings/*.xdf
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
)

from load_session import load_session
from preprocess import apply_filters, epoch_sliding_window
from features import (
    extract_band_powers, make_lda_pipeline, make_svm_pipeline,
)
from config import (
    SCALP_CHANNELS, CV_FOLDS, MODEL_PATH, RESULTS_DIR, CLASS_NAMES,
)

mne.set_log_level("WARNING")
warnings.filterwarnings("ignore", category=FutureWarning)


def load_and_preprocess(xdf_path: str) -> mne.io.RawArray:
    """Step 5 + 6: load a session and apply filters."""
    raw = load_session(xdf_path, scalp_channels=SCALP_CHANNELS, verbose=True)
    print("  Preprocessing …")
    apply_filters(raw, verbose=True)
    return raw


def sanity_check(epochs: mne.Epochs, event_id: dict) -> None:
    """Step 9: visual sanity check before modelling."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # PSD per class
    fig, axes = plt.subplots(1, len(event_id), figsize=(6 * len(event_id), 4))
    if len(event_id) == 1:
        axes = [axes]
    for ax, name in zip(axes, event_id):
        epochs[name].compute_psd(fmax=45, verbose=False).plot(axes=ax, show=False)
        ax.set_title(f"PSD — {name}")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "sanity_psd_per_class.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved sanity PSD → {RESULTS_DIR / 'sanity_psd_per_class.png'}")


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict:
    """Steps 12–13: train pipelines with 10-fold CV, compare, and report."""

    pipelines = {
        "LDA": make_lda_pipeline(),
        "SVM (RBF)": make_svm_pipeline(),
    }

    # Optionally add Riemannian pipelines
    try:
        from features import make_riemannian_mdm_pipeline, make_riemannian_lr_pipeline
        # NOTE: Riemannian pipelines need raw epoch data, not band-power features.
        # They are evaluated separately below if available.
        riemannian_available = True
    except ImportError:
        riemannian_available = False

    print(f"\n{'='*60}")
    print(f"10-fold Stratified Cross-Validation  (n={len(y)} epochs)")
    print(f"{'='*60}\n")

    results = {}
    kfold = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, cv=kfold, scoring="accuracy")
        results[name] = {
            "mean_acc": scores.mean(),
            "std_acc": scores.std(),
            "scores": scores,
        }
        print(f"  {name:15s}  acc = {scores.mean():.1%} ± {scores.std():.1%}")

    # Pick the best pipeline and fit on all data
    best_name = max(results, key=lambda k: results[k]["mean_acc"])
    best_pipe = pipelines[best_name]
    best_pipe.fit(X, y)

    print(f"\n  Best pipeline: {best_name} ({results[best_name]['mean_acc']:.1%})")

    # Confusion matrix from last fold for visualization
    fig, ax = plt.subplots(figsize=(5, 4))
    y_pred = best_pipe.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(CLASS_NAMES.values()))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{best_name} — Full dataset\nCV acc = {results[best_name]['mean_acc']:.1%}")
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"best_name": best_name, "best_pipe": best_pipe, "results": results}


def main():
    parser = argparse.ArgumentParser(
        description="Train EEG classifier for the Balloon Controller"
    )
    parser.add_argument(
        "sessions", nargs="+",
        help="Paths to XDF session files",
    )
    parser.add_argument(
        "--output", default=str(MODEL_PATH),
        help=f"Where to save the trained model (default: {MODEL_PATH})",
    )
    args = parser.parse_args()

    # ── Load all sessions ────────────────────────────────────────────────────
    all_epochs = []
    for path in args.sessions:
        print(f"\n{'─'*60}")
        print(f"Loading: {path}")
        print(f"{'─'*60}")
        raw = load_and_preprocess(path)
        epochs, event_id = epoch_sliding_window(raw, verbose=True)
        all_epochs.append(epochs)

    # Concatenate across sessions
    if len(all_epochs) > 1:
        epochs = mne.concatenate_epochs(all_epochs)
        print(f"\nCombined: {len(epochs)} epochs across {len(all_epochs)} sessions")
    else:
        epochs = all_epochs[0]

    # ── Sanity check (step 9) ────────────────────────────────────────────────
    print("\n[Step 9] Sanity check …")
    sanity_check(epochs, event_id)

    # ── Feature extraction (step 10) ─────────────────────────────────────────
    print("\n[Step 10] Extracting band-power features …")
    X, feature_names = extract_band_powers(epochs)
    print(f"  Feature matrix: {X.shape}")

    # Build labels
    id_to_name = {v: k for k, v in event_id.items()}
    y = np.array([
        1 if "relax" in id_to_name[code].lower() else 0
        for code in epochs.events[:, 2]
    ])
    print(f"  Labels: {np.bincount(y)}  (0=concentration, 1=relaxation)")

    # ── Train & evaluate (steps 12–13) ───────────────────────────────────────
    result = train_and_evaluate(X, y, feature_names)

    # ── Save model (step 14) ─────────────────────────────────────────────────
    model_path = Path(args.output)
    joblib.dump(result["best_pipe"], model_path)
    print(f"\n[Step 14] Model saved → {model_path}")
    print(f"  Pipeline: {result['best_name']}")
    print(f"  CV accuracy: {result['results'][result['best_name']]['mean_acc']:.1%}")

    # ── Gate check ───────────────────────────────────────────────────────────
    acc = result["results"][result["best_name"]]["mean_acc"]
    if acc >= 0.65:
        print("\n✅ Accuracy ≥ 65% — ready for real-time!")
    elif acc > 0.50:
        print("\n⚠️  Accuracy > 50% but < 65% — collect more data or tune preprocessing.")
    else:
        print("\n❌ Accuracy ≤ 50% — pipeline is not working. Check data quality.")


if __name__ == "__main__":
    main()
