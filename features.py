"""
Feature extraction — Steps 10–11 of the project spec.

Two feature pipelines:
  1. Band-power features (baseline)
  2. Riemannian geometry features (advanced, via PyRiemann)
"""

from __future__ import annotations

import numpy as np
import mne
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from config import BANDS


# ─────────────────────────────────────────────────────────────────────────────
# Step 10 — Band-power features
# ─────────────────────────────────────────────────────────────────────────────

def extract_band_powers(epochs: mne.Epochs) -> tuple[np.ndarray, list[str]]:
    """Compute average power per band per channel for each epoch.

    Returns
    -------
    X : (n_epochs, n_channels * n_bands)
    feature_names : list of human-readable feature names
    """
    all_powers = []
    feature_names = []

    for band_name, (fmin, fmax) in BANDS.items():
        psd = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose=False)
        power = psd.get_data()              # (n_epochs, n_channels, n_freqs)
        band_power = power.mean(axis=2)     # (n_epochs, n_channels)
        all_powers.append(band_power)
        for ch in epochs.ch_names:
            feature_names.append(f"{ch}_{band_name}")

    X = np.concatenate(all_powers, axis=1)
    return X, feature_names


def extract_band_powers_from_array(
    data: np.ndarray, sfreq: float
) -> np.ndarray:
    """Extract band powers from a raw numpy array (for real-time use).

    Parameters
    ----------
    data : (n_channels, n_times) — one window of EEG
    sfreq : sampling frequency

    Returns
    -------
    features : (1, n_channels * n_bands) — ready for model.predict()
    """
    from scipy.signal import welch

    features = []
    for _band_name, (fmin, fmax) in BANDS.items():
        freqs, pxx = welch(data, fs=sfreq, nperseg=min(data.shape[1], int(sfreq)))
        # Select frequency bins within this band
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = pxx[:, band_mask].mean(axis=1)  # (n_channels,)
        features.append(band_power)

    return np.concatenate(features).reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 11 — Riemannian features (optional, requires pyriemann)
# ─────────────────────────────────────────────────────────────────────────────

def make_riemannian_mdm_pipeline():
    """MDM classifier in Riemannian space (no tangent projection).

    Requires: pip install pyriemann
    """
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.classification import MDM
    except ImportError:
        raise ImportError(
            "pyriemann is required for Riemannian features. "
            "Install with: pip install pyriemann"
        )
    return make_pipeline(Covariances("oas"), MDM())


def make_riemannian_lr_pipeline():
    """TangentSpace + LogisticRegression (often beats MDM).

    Requires: pip install pyriemann
    """
    try:
        from pyriemann.estimation import Covariances
        from pyriemann.tangentspace import TangentSpace
    except ImportError:
        raise ImportError(
            "pyriemann is required for Riemannian features. "
            "Install with: pip install pyriemann"
        )
    return make_pipeline(
        Covariances("oas"),
        TangentSpace(),
        LogisticRegression(max_iter=1000),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standard sklearn pipelines (band-power based)
# ─────────────────────────────────────────────────────────────────────────────

def make_lda_pipeline():
    """StandardScaler → LDA."""
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())


def make_svm_pipeline():
    """StandardScaler → SVM (RBF kernel)."""
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True))
