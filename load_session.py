"""
Canonical EEG session loader — XDF → MNE Raw.

Every notebook and script in this project should import this module
instead of re-implementing XDF parsing.  The single public function
``load_session`` takes a file path and returns a ready-to-use MNE
`~mne.io.RawArray` with trial-start annotations already attached.

MNE API surface used (reference: https://mne.tools/stable/generated/mne.io.Raw.html):

* ``mne.create_info``   — build channel metadata
* ``mne.io.RawArray``   — construct Raw from (n_channels, n_times) numpy array
* ``raw.set_annotations``— attach ``mne.Annotations`` to the Raw
* ``raw[picks, slice]``  — ``__getitem__`` for channel / time indexing
* ``raw.info``           — dict-like access to sfreq, ch_names, etc.

Typical usage
-------------
>>> from load_session import load_session
>>> raw = load_session("sub-P001_ses-S001_task-Default_run-001_eeg.xdf")
>>> raw.info["sfreq"]
256.0
>>> raw.annotations          # trial-start markers
>>> data, times = raw[:]     # __getitem__ → (n_channels, n_times), (n_times,)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pyxdf
import mne

__all__ = ["load_session"]

logger = logging.getLogger(__name__)

# Default scalp channels expected from the MuseLSL / OpenBCI cap
_DEFAULT_SCALP_CHANNELS = ("F3", "F4", "C3", "Cz", "C4", "P3", "P4")


def load_session(
    xdf_path: Union[str, Path],
    scalp_channels: Sequence[str] = _DEFAULT_SCALP_CHANNELS,
    *,
    marker_suffix: str = "_start",
    verbose: bool = True,
) -> mne.io.RawArray:
    """Load an XDF session file and return an MNE Raw object with annotations.

    Parameters
    ----------
    xdf_path : str | Path
        Path to the ``.xdf`` recording file.
    scalp_channels : sequence of str, optional
        Ordered list of channel labels to retain.  If the XDF metadata does
        not contain matching labels, the first ``len(scalp_channels)``
        channels are used instead (with generic names).
    marker_suffix : str, optional
        Only marker strings ending with this suffix are injected as MNE
        annotations.  Set to ``""`` to keep *all* markers.
    verbose : bool, optional
        Print a loading summary to stdout.

    Returns
    -------
    mne.io.RawArray
        Continuous EEG data with ``mne.Annotations`` attached.  Data is
        scaled to Volts (MNE convention).  Access the underlying array via
        ``raw[:]`` (calls ``__getitem__``), query metadata via
        ``raw.info``, and read trial markers via ``raw.annotations``.

    Raises
    ------
    FileNotFoundError
        If *xdf_path* does not exist.
    RuntimeError
        If no EEG or marker stream can be identified in the file.

    Examples
    --------
    >>> raw = load_session("recording.xdf")
    >>> raw.info["sfreq"]
    256.0
    >>> data, times = raw[:]          # shape (n_ch, n_times), (n_times,)
    >>> raw.annotations.description   # array of marker strings
    """
    xdf_path = Path(xdf_path)
    if not xdf_path.exists():
        raise FileNotFoundError(f"XDF file not found: {xdf_path}")

    # ── 1. Parse the XDF container ──────────────────────────────────────────
    streams, _header = pyxdf.load_xdf(str(xdf_path))
    eeg_stream, marker_stream = _identify_streams(streams)

    sfreq = float(eeg_stream["info"]["nominal_srate"][0])
    n_recorded = int(eeg_stream["info"]["channel_count"][0])
    eeg_data   = eeg_stream["time_series"]      # (n_samples, n_channels)
    eeg_times  = eeg_stream["time_stamps"]       # (n_samples,)

    marker_strings = [m[0] for m in marker_stream["time_series"]]
    marker_times   = marker_stream["time_stamps"]

    # ── 2. Resolve channel names & indices ──────────────────────────────────
    all_ch_names = _extract_channel_names(eeg_stream, n_recorded)
    ch_names, use_indices = _resolve_channels(
        all_ch_names, list(scalp_channels), n_recorded, verbose,
    )

    # ── 3. Build the MNE Raw object ─────────────────────────────────────────
    scalp_data = eeg_data[:, use_indices]              # (n_samples, n_ch)
    scale = _infer_scale(scalp_data, verbose)

    info = mne.create_info(
        ch_names=ch_names,
        sfreq=sfreq,
        ch_types=["eeg"] * len(ch_names),
    )
    raw = mne.io.RawArray((scalp_data * scale).T, info, verbose=False)

    # ── 4. Inject markers as MNE Annotations ────────────────────────────────
    raw_start = eeg_times[0]
    onsets = marker_times - raw_start

    if marker_suffix:
        mask = [m.endswith(marker_suffix) for m in marker_strings]
        annot_onsets       = onsets[mask]
        annot_descriptions = [m for m, keep in zip(marker_strings, mask) if keep]
    else:
        annot_onsets       = onsets
        annot_descriptions = marker_strings

    annotations = mne.Annotations(
        onset=annot_onsets,
        duration=[0.0] * len(annot_onsets),
        description=annot_descriptions,
    )
    raw.set_annotations(annotations)

    # ── 5. Summary ──────────────────────────────────────────────────────────
    if verbose:
        print(f"[load_session] {xdf_path.name}")
        print(f"  Channels  : {raw.info['nchan']}  {raw.ch_names}")
        print(f"  Sfreq     : {sfreq} Hz")
        print(f"  Duration  : {raw.times[-1]:.1f} s  ({raw.n_times} samples)")
        print(f"  Annotations: {len(raw.annotations)} markers")

    return raw


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _identify_streams(streams: list) -> tuple:
    """Find the EEG and marker streams in an XDF file.

    Strategy: first try declared ``type`` metadata, then fall back to a
    channel-count heuristic (multi-channel → EEG, single-channel → marker).
    """
    eeg_stream: Optional[dict] = None
    marker_stream: Optional[dict] = None

    # Primary: declared stream type
    for s in streams:
        stype = s["info"]["type"][0].lower()
        if stype in ("eeg", "data"):
            eeg_stream = s
        elif stype in ("markers", "marker", "events"):
            marker_stream = s

    # Fallback: channel-count heuristic
    if eeg_stream is None or marker_stream is None:
        for s in streams:
            n_ch = int(s["info"]["channel_count"][0])
            if n_ch > 1 and eeg_stream is None:
                eeg_stream = s
            elif n_ch == 1 and marker_stream is None:
                marker_stream = s

    if eeg_stream is None:
        raise RuntimeError("Could not identify an EEG stream in the XDF file.")
    if marker_stream is None:
        raise RuntimeError("Could not identify a marker stream in the XDF file.")

    return eeg_stream, marker_stream


def _extract_channel_names(eeg_stream: dict, n_channels: int) -> list[str]:
    """Read channel labels from XDF metadata, or generate CH1…CHn fallback."""
    try:
        ch_info = eeg_stream["info"]["desc"][0]["channels"][0]["channel"]
        names = [ch["label"][0] for ch in ch_info]
        if len(names) == n_channels:
            return names
    except (KeyError, TypeError, IndexError):
        pass

    return [f"CH{i + 1}" for i in range(n_channels)]


def _resolve_channels(
    all_names: list[str],
    wanted: list[str],
    n_recorded: int,
    verbose: bool,
) -> tuple[list[str], list[int]]:
    """Map requested channel labels → indices, with a first-N fallback."""
    lookup = {name: idx for idx, name in enumerate(all_names)}
    missing = [name for name in wanted if name not in lookup]

    if missing:
        if verbose:
            logger.warning(
                "Channels %s not in XDF metadata; using first %d channels.",
                missing, len(wanted),
            )
        n = min(len(wanted), n_recorded)
        indices = list(range(n))
        names = [all_names[i] for i in indices]
    else:
        indices = [lookup[name] for name in wanted]
        names = [all_names[i] for i in indices]

    return names, indices


def _infer_scale(data: np.ndarray, verbose: bool) -> float:
    """Guess whether data is in µV (scale → 1e-6) or already in V."""
    p99 = np.nanpercentile(np.abs(data), 99)
    if p99 > 1e-3:
        if verbose:
            print(f"  Scale     : µV → V  (99th-percentile abs = {p99:.3g})")
        return 1e-6
    if verbose:
        print(f"  Scale     : already in V  (99th-percentile abs = {p99:.3g})")
    return 1.0
