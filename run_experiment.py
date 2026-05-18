"""
LSL Experiment Cue Script — Step 2 of the project spec.

Plays audio/visual cues for the EEG experiment and pushes event markers
onto a separate LSL stream. Record both the EEG and marker streams
together using LabRecorder.

Protocol:
    10s concentration → 4s rest → 10s relaxation → 4s rest → repeat

Usage:
    python run_experiment.py                     # 40 trials per class
    python run_experiment.py --trials 20         # fewer trials
    python run_experiment.py --start-number 873  # custom subtraction start
"""

from __future__ import annotations

import argparse
import random
import sys
import time


def run_experiment(n_trials_per_class: int = 40, start_number: int | None = None):
    """Run the concentration vs relaxation cue protocol."""
    try:
        from pylsl import StreamInfo, StreamOutlet
    except ImportError:
        print("ERROR: pylsl is required to run the experiment.")
        print("Install with: pip install pylsl")
        sys.exit(1)

    # Create LSL marker stream
    info = StreamInfo(
        name="ExperimentMarkers",
        type="Markers",
        channel_count=1,
        nominal_srate=0,    # irregular rate (event-based)
        channel_format="string",
        source_id="eeg_balloon_markers",
    )
    outlet = StreamOutlet(info)

    # Generate random starting number for mental subtraction
    if start_number is None:
        start_number = random.randint(700, 999)

    # Build trial sequence: alternating concentration / relaxation
    # Randomize to avoid order effects
    total_trials = n_trials_per_class * 2
    trial_types = (["concentration"] * n_trials_per_class +
                   ["relaxation"] * n_trials_per_class)
    random.shuffle(trial_types)

    trial_duration = 10.0   # seconds per trial
    rest_duration = 4.0     # seconds between trials (blink/rest period)

    total_time = total_trials * (trial_duration + rest_duration)

    print("=" * 60)
    print("  EEG Balloon Controller — Experiment")
    print("=" * 60)
    print(f"\n  Trials: {n_trials_per_class} per class ({total_trials} total)")
    print(f"  Trial duration: {trial_duration}s")
    print(f"  Rest between trials: {rest_duration}s")
    print(f"  Estimated total time: {total_time/60:.1f} minutes")
    print(f"\n  Starting subtraction number: {start_number}")
    print(f"  (Subtract 7 repeatedly: {start_number}, {start_number-7}, "
          f"{start_number-14}, …)")
    print("\n  INSTRUCTIONS:")
    print("  - CONCENTRATION: Silently subtract 7 continuously")
    print("  - RELAXATION: Close your eyes, breathe naturally, relax")
    print("  - REST periods: Blink freely, stay still")
    print()

    input("  Press ENTER when LabRecorder is recording and subject is ready …")
    print()

    # Countdown
    for i in range(3, 0, -1):
        print(f"  Starting in {i} …")
        time.sleep(1.0)
    print()

    # Push session start marker
    outlet.push_sample(["session_start"])

    for trial_num, trial_type in enumerate(trial_types, 1):
        marker = f"{trial_type}_start"

        # Print cue
        if trial_type == "concentration":
            print(f"  Trial {trial_num}/{total_trials}: "
                  f"🧮 CONCENTRATE — subtract 7 from {start_number}")
        else:
            print(f"  Trial {trial_num}/{total_trials}: "
                  f"😌 RELAX — close eyes, breathe naturally")

        # Push marker
        outlet.push_sample([marker])

        # Wait for trial duration
        time.sleep(trial_duration)

        # End marker
        outlet.push_sample([f"{trial_type}_end"])

        # Rest period
        if trial_num < total_trials:
            print(f"  {'─'*40}  Rest ({rest_duration}s)")
            outlet.push_sample(["rest_start"])
            time.sleep(rest_duration)

    # Session end
    outlet.push_sample(["session_end"])
    print(f"\n{'='*60}")
    print("  Session complete! Stop LabRecorder now.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="EEG Experiment Cue Script")
    parser.add_argument(
        "--trials", type=int, default=40,
        help="Number of trials per class (default: 40)",
    )
    parser.add_argument(
        "--start-number", type=int, default=None,
        help="Starting number for subtraction (random if not given)",
    )
    args = parser.parse_args()
    run_experiment(args.trials, args.start_number)


if __name__ == "__main__":
    main()
