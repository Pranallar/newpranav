"""
Game bridge — Step 18 of the project spec.

Standalone bridge that reads classifier output and forwards commands
to the UChicago Neurotech balloon_control_game.

The classifier publishes commands over UDP. This bridge reads them
and translates to whatever the game expects (keyboard simulation,
file-based, or direct API call).

Architecture:
    [Classifier] --UDP--> [game_bridge.py] ---> [Balloon Game]

The game doesn't import any ML code.

Usage:
    python game_bridge.py               # listens on UDP 5555, prints commands
    python game_bridge.py --keyboard    # simulates keyboard presses
"""

from __future__ import annotations

import argparse
import json
import socket
import sys

from config import GAME_HOST, GAME_PORT


def listen_and_forward(keyboard_mode: bool = False):
    """Listen for classifier commands on UDP and forward to game."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((GAME_HOST, GAME_PORT))
    sock.settimeout(1.0)

    print(f"Game bridge listening on {GAME_HOST}:{GAME_PORT}")
    print(f"Mode: {'keyboard simulation' if keyboard_mode else 'print only'}")
    print("Waiting for classifier commands …\n")

    # Map classifier output → game action
    COMMAND_MAP = {
        "concentration": "inflate",    # math = focus = inflate balloon
        "relaxation":    "deflate",    # rest = relax  = deflate balloon
    }

    try:
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                msg = json.loads(data.decode("utf-8"))
                command = msg.get("command", "")
                action = COMMAND_MAP.get(command, "none")

                print(f"  Received: {command:15s} → Action: {action}")

                if keyboard_mode and action != "none":
                    _simulate_key(action)

            except socket.timeout:
                continue

    except KeyboardInterrupt:
        print("\nBridge stopped.")
    finally:
        sock.close()


def _simulate_key(action: str):
    """Simulate a keyboard press for the balloon game.

    Requires: pip install pynput
    """
    try:
        from pynput.keyboard import Controller, Key
        keyboard = Controller()

        if action == "inflate":
            keyboard.press(Key.up)
            keyboard.release(Key.up)
        elif action == "deflate":
            keyboard.press(Key.down)
            keyboard.release(Key.down)
    except ImportError:
        pass  # silently skip if pynput not installed


def main():
    parser = argparse.ArgumentParser(description="Balloon Game Bridge")
    parser.add_argument(
        "--keyboard", action="store_true",
        help="Simulate keyboard presses (requires pynput)",
    )
    args = parser.parse_args()
    listen_and_forward(args.keyboard)


if __name__ == "__main__":
    main()
