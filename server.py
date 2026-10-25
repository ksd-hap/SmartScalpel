"""SmartScalpel Theatre server. SIMULATION ONLY - not a clinical tool.

Runs on the laptop. The iPad opens the theatre page over Wi-Fi; every burn is classified here by the
real Python model (app.ml_service.predict_tissue), and the LED colour is sent to the Arduino over USB.

    py -m pip install -r requirements-web.txt
    py server.py                     # auto-detects the Arduino
    py server.py --serial COM5       # or name the port
    py server.py --no-serial         # page + model only

Then open the address it prints (http://<laptop-ip>:5000) in Safari on the iPad.
Laptop and iPad must be on the same network. Venue Wi-Fi (e.g. eduroam) often blocks device-to-device
traffic; if the iPad can't connect, use a phone hotspot or Windows "Mobile hotspot" for both devices.
"""
from __future__ import annotations

import argparse
import socket
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from app.ml_service import load_model, predict_tissue

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DEFAULT_CSV = ROOT / "data" / "spectra_input.csv"
LED_STATES = {"RED", "YELLOW", "GREEN", "OFF"}
# Words understood by the team's Mega sketch (exact lines, no counter). Its own colours:
# tumour -> TUMOUR (red), adipose -> ADIPOSE, stroma -> YELLOW, off -> OFF.
SKETCH_CMD = {"RED": "TUMOUR", "GREEN": "ADIPOSE", "YELLOW": "YELLOW", "OFF": "OFF"}  # stroma shows yellow
HEARTBEAT_S = 0.5  # the sketch switches the LED off if no command arrives for 2 s

app = Flask(__name__, static_folder=None)
SPECTRA: dict[int, np.ndarray] = {}


# ---------------------------------------------------------------- Arduino over USB serial
class Led:
    """Sends 'TUMOUR\\n' / 'ADIPOSE\\n' / 'OFF\\n' lines at 115200 baud to the team's Mega sketch,
    repeating the current colour every 0.5 s because the sketch turns the LED off after 2 s of silence."""

    def __init__(self, port: str | None, enabled: bool):
        self.wanted, self.enabled = port, enabled
        self.ser, self.port, self.last = None, None, None
        self.lock = threading.Lock()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        threading.Thread(target=self._reader, daemon=True).start()

    def _write(self, state: str) -> bool:
        try:
            self.ser.write(f"{SKETCH_CMD.get(state, state)}\n".encode())
            return True
        except Exception as e:  # noqa: BLE001
            print(f"  Serial write failed: {e}")
            self.ser = None
            return False

    def _heartbeat(self) -> None:
        while True:
            time.sleep(HEARTBEAT_S)
            with self.lock:
                if self.ser and self.last and self.last != "OFF":
                    self._write(self.last)

    def _reader(self) -> None:
        """Print the sketch's replies that matter (errors, timeouts, READY); ACKs are skipped."""
        while True:
            ser = self.ser
            if not ser:
                time.sleep(.3)
                continue
            try:
                line = ser.readline().decode(errors="replace").strip()
            except Exception:  # noqa: BLE001
                time.sleep(.3)
                continue
            if line and not line.startswith("ACK:"):
                print(f"  Arduino says: {line}")

    def connect(self) -> None:
        if not self.enabled:
            return
        try:
            import serial
            from serial.tools import list_ports
        except ImportError:
            print("  pyserial not installed - LED disabled (py -m pip install pyserial)")
            return
        with self.lock:
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None
            port = self.wanted
            if not port:
                ports = list(list_ports.comports())
                hits = [p for p in ports if any(k in f"{p.description} {p.manufacturer}".lower()
                                               for k in ("arduino", "ch340", "wch", "usb serial", "usb-serial"))]
                port = (hits or ports or [None])[0]
                port = port.device if port else None
            if not port:
                print("  No serial port found - plug in the Arduino and press 'Check Arduino' on the page")
                return
            try:
                ser = serial.Serial(port, 115200, timeout=1)
                time.sleep(2.0)  # the Mega resets when the port opens, then prints READY
                self.ser, self.port, self.last = ser, port, None
                print(f"  Arduino connected on {port}")
            except Exception as e:  # noqa: BLE001
                print(f"  Could not open {port}: {e}")
                self.ser = None

    def send(self, state: str) -> bool:
        with self.lock:
            if not self.ser:
                return False
            if state == self.last:
                return True
            ok = self._write(state)
            if ok:
                self.last = state
            return ok

    @property
    def status(self) -> dict:
        return {"connected": self.ser is not None, "port": self.port}


led: Led

# LED state shared with led_bridge.py on another laptop (long-polling over the network)
LED_NOW = {"state": "OFF", "seq": 0}
LED_COND = threading.Condition()
BRIDGE = {"seen": 0.0, "port": None, "host": None}


def publish_led(state: str) -> None:
    with LED_COND:
        if LED_NOW["state"] != state:
            LED_NOW["state"] = state
            LED_NOW["seq"] += 1
            LED_COND.notify_all()


def serial_status() -> dict:
    if led.status["connected"]:
        return {**led.status, "where": "this laptop"}
    if BRIDGE["port"] and time.time() - BRIDGE["seen"] < 40:
        return {"connected": True, "port": f"{BRIDGE['port']} on {BRIDGE['host']}", "where": "bridge"}
    return {"connected": False, "port": None, "where": None}


# ---------------------------------------------------------------- routes
@app.get("/")
@app.get("/theatre")
def index():
    resp = send_from_directory(WEB, "index.html", max_age=0)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.get("/api/health")
def health():
    m = load_model()
    page = WEB / "index.html"
    return jsonify(service="SmartScalpel-Theatre", model_version=m["model_version"], spectra=len(SPECTRA),
                   serial=serial_status(), page=str(page), page_modified=time.ctime(page.stat().st_mtime))


@app.post("/api/classify")
def classify():
    body = request.get_json(silent=True) or {}
    try:
        sid = int(body["sample_id"])
        spectrum = SPECTRA[sid]
    except (KeyError, TypeError, ValueError):
        return jsonify(error="unknown sample_id"), 422
    t0 = time.perf_counter()
    result = predict_tissue(spectrum)
    result["ms"] = round((time.perf_counter() - t0) * 1000, 2)
    result["sample_id"] = sid
    return jsonify(result)


@app.post("/api/led")
def set_led():
    state = str((request.get_json(silent=True) or {}).get("state", "")).upper()
    if state not in LED_STATES:
        return jsonify(error=f"state must be one of {sorted(LED_STATES)}"), 422
    publish_led(state)
    led.send(state)
    return jsonify(ok=True, state=state, serial=serial_status())


@app.get("/api/led/next")
def led_next():
    """Long-poll used by led_bridge.py: returns as soon as the LED state changes (or after ~25 s)."""
    after = request.args.get("after", type=int, default=-1)
    BRIDGE.update(seen=time.time(), port=request.args.get("port") or None, host=request.remote_addr)
    with LED_COND:
        LED_COND.wait_for(lambda: LED_NOW["seq"] != after, timeout=25)
        return jsonify(LED_NOW)


@app.post("/api/serial/reconnect")
def reconnect():
    led.connect()
    led.send("OFF")
    return jsonify(serial=serial_status())


# ---------------------------------------------------------------- startup
def lan_addresses() -> list[str]:
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the outgoing interface
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def port_free(port: int) -> bool:
    """True if nothing else is listening on this port (checks both all-interfaces and localhost)."""
    for host in ("0.0.0.0", "127.0.0.1"):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):  # Windows: refuse to share a port another app holds
                s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            s.bind((host, port))
        except OSError:
            return False
        finally:
            s.close()
    return True


def main() -> None:
    global led
    ap = argparse.ArgumentParser(description="SmartScalpel Theatre server")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="held-out spectra CSV (sample_id + 8000 m/z columns)")
    ap.add_argument("--serial", default=None, help="Arduino port, e.g. COM5 or /dev/ttyACM0 (default: auto-detect)")
    ap.add_argument("--no-serial", action="store_true", help="run without the Arduino LED")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()

    print("SmartScalpel Theatre")
    m = load_model()
    df = pd.read_csv(args.csv)
    X = df.drop(columns="sample_id").to_numpy(dtype=np.float64)
    if X.shape[1] != m["n_features"]:
        raise SystemExit(f"CSV has {X.shape[1]} features, model expects {m['n_features']}")
    SPECTRA.update({int(s): x for s, x in zip(df["sample_id"], X)})
    print(f"  Model {m['model_version']} loaded · {len(SPECTRA)} spectra from {args.csv}")
    if not (WEB / "index.html").exists():
        raise SystemExit(f"Missing {WEB / 'index.html'}")

    led = Led(args.serial, not args.no_serial)
    led.connect()
    led.send("OFF")
    if not led.status["connected"]:
        print("  (If the Arduino is on another laptop, run led_bridge.bat there - see led_bridge.py)")

    page = WEB / "index.html"
    print(f"  Serving {page} (saved {time.ctime(page.stat().st_mtime)})")
    port = args.port
    while not port_free(port):
        print(f"  Port {port} is already used by another program (an older server?) - trying {port + 1}")
        port += 1
    args.port = port

    print("\nOpen one of these in Safari on the iPad (same Wi-Fi as this laptop):")
    for ip in lan_addresses() or ["<this laptop's IP>"]:
        print(f"    http://{ip}:{args.port}/theatre")
    print(f"On this laptop: http://localhost:{args.port}/theatre\n")
    app.run(host="0.0.0.0", port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
