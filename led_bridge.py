"""SmartScalpel LED bridge - run this on the laptop the Arduino is plugged into.

It connects to the SmartScalpel Theatre server on another laptop over the network, waits for LED
changes, and forwards them to the Arduino over USB as the team sketch's words ("TUMOUR", "ADIPOSE",
"OFF" lines at 115200 baud), repeating the colour every 0.5 s because the sketch turns off after 2 s.

    py -m pip install pyserial
    py led_bridge.py --server http://192.168.1.23:5000            # auto-detects the Arduino
    py led_bridge.py --server http://192.168.1.23:5000 --serial COM5
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request

import serial
from serial.tools import list_ports

SKETCH_CMD = {"RED": "TUMOUR", "GREEN": "ADIPOSE", "YELLOW": "YELLOW", "OFF": "OFF"}  # stroma shows yellow
STATE = {"ser": None, "cmd": "OFF"}
LOCK = threading.Lock()


def heartbeat() -> None:
    while True:
        time.sleep(0.5)
        with LOCK:
            if STATE["ser"] and STATE["cmd"] != "OFF":
                try:
                    STATE["ser"].write(f"{STATE['cmd']}\n".encode())
                except Exception:  # noqa: BLE001
                    pass


def find_port(wanted: str | None) -> str | None:
    if wanted:
        return wanted
    ports = list(list_ports.comports())
    hits = [p for p in ports if any(k in f"{p.description} {p.manufacturer}".lower()
                                   for k in ("arduino", "ch340", "wch", "usb serial", "usb-serial"))]
    for p in ports:
        print(f"    found {p.device}: {p.description}")
    pick = (hits or ports or [None])[0]
    return pick.device if pick else None


def open_serial(wanted: str | None):
    while True:
        port = find_port(wanted)
        if port:
            try:
                ser = serial.Serial(port, 115200, timeout=1)
                time.sleep(2.0)  # the Mega resets when the port opens
                print(f"Arduino connected on {port}")
                return ser, port
            except Exception as e:  # noqa: BLE001
                print(f"Could not open {port}: {e} (close the Arduino IDE Serial Monitor?)")
        else:
            print("No Arduino found - plug it in. Retrying in 3 s...")
        time.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Forward SmartScalpel LED colours to a local Arduino")
    ap.add_argument("--server", required=True, help="address of the theatre server, e.g. http://192.168.1.23:5000")
    ap.add_argument("--serial", default=None, help="Arduino port, e.g. COM5 (default: auto-detect)")
    args = ap.parse_args()
    base = args.server.rstrip("/")
    if not base.startswith("http"):
        base = "http://" + base

    ser, port = open_serial(args.serial)
    STATE["ser"] = ser
    threading.Thread(target=heartbeat, daemon=True).start()
    seq = -1
    print(f"Listening to {base} ... (Ctrl+C to stop)")
    while True:
        try:
            with urllib.request.urlopen(f"{base}/api/led/next?after={seq}&port={port}", timeout=35) as r:
                msg = json.load(r)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"Can't reach {base} ({e}). Same Wi-Fi? Server running? Retrying in 2 s...")
            time.sleep(2)
            continue
        if msg["seq"] == seq:
            continue  # timeout with no change; poll again
        seq = msg["seq"]
        cmd = SKETCH_CMD.get(msg["state"], msg["state"])
        try:
            with LOCK:
                STATE["cmd"] = cmd
                ser.write(f"{cmd}\n".encode())
            print(f"LED -> {cmd}")
        except Exception as e:  # noqa: BLE001
            print(f"Arduino write failed ({e}); reconnecting...")
            ser, port = open_serial(args.serial)
            with LOCK:
                STATE["ser"] = ser
            seq = -1  # resend the current state


if __name__ == "__main__":
    main()
