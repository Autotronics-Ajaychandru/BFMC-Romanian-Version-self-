"""
dashboard/web_dashboard.py
==========================
Lightweight Flask web dashboard for the BFMC 2026 autonomous stack.

Usage (from main.py):
    from dashboard.web_dashboard import WebDashboard
    web = WebDashboard()
    web.start()                        # starts Flask in daemon thread
    web.push_telemetry(**state_dict)   # call every control loop tick
    web.push_frame(bgr_frame)          # call with the camera frame
    cmds = web.pop_commands()          # list of {"action": ..., "value": ...}

Routes
------
  GET  /              → Full dashboard HTML page
  GET  /stream        → MJPEG camera stream
  GET  /api/state     → JSON telemetry snapshot
  GET  /api/log       → JSON array of recent log lines
  POST /api/command   → {"action": str, "value": optional}
"""

import io
import json
import threading
import time
from collections import deque
from typing import Any, Dict, List

import cv2
import numpy as np

from config import WEB_DASHBOARD_HOST, WEB_DASHBOARD_PORT, WEB_DASHBOARD_FPS

try:
    from flask import Flask, Response, jsonify, request
    _FLASK_AVAILABLE = True
except ImportError:
    _FLASK_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
#  Module-level shared state (thread-safe via locks)
# ─────────────────────────────────────────────────────────────────────────────
_state: Dict[str, Any] = {
    "mode": "MANUAL",
    "speed_pwm": 0,
    "steer_deg": 0.0,
    "yaw_deg": 0.0,
    "roll_deg": 0.0,
    "pitch_deg": 0.0,
    "car_x": 0.0,
    "car_y": 0.0,
    "lane_anchor": "—",
    "target_x": 320.0,
    "lateral_err_px": 0.0,
    "lane_confidence": 0.0,
    "active_sign": "—",
    "yolo_labels": [],
    "battery_pct": 0,
    "loop_hz": 0.0,
    "is_recording": False,
    "base_speed": 150.0,
    "steer_mult": 1.0,
    "sign_detect_m": 5.0,
    "sign_act_m": 2.0,
}
_state_lock = threading.Lock()

_latest_frame: bytes = b""
_frame_lock = threading.Lock()
_frame_event = threading.Event()

_log_lines: deque = deque(maxlen=100)
_log_lock = threading.Lock()

_pending_commands: List[Dict] = []
_cmd_lock = threading.Lock()

# ─────────────────────────────────────────────────────────────────────────────
#  Public API (called from main.py control loop — must be non-blocking)
# ─────────────────────────────────────────────────────────────────────────────

def push_telemetry(**kwargs) -> None:
    with _state_lock:
        _state.update(kwargs)

def push_frame(bgr_frame: np.ndarray) -> None:
    _, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    with _frame_lock:
        global _latest_frame
        _latest_frame = buf.tobytes()
    _frame_event.set()

def push_log(message: str, level: str = "INFO") -> None:
    ts = time.strftime("%H:%M:%S")
    with _log_lock:
        _log_lines.append({"ts": ts, "level": level, "msg": message})

def pop_commands() -> List[Dict]:
    with _cmd_lock:
        cmds = list(_pending_commands)
        _pending_commands.clear()
    return cmds


# ─────────────────────────────────────────────────────────────────────────────
#  Embedded HTML (single-file, no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BFMC 2026 — Remote Dashboard</title>
<style>
  :root {
    --bg: #1e1e1e; --panel: #252526; --fg: #cccccc;
    --accent: #007acc; --danger: #f44336; --success: #4caf50;
    --warn: #ff9800; --border: #333;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--fg); font-family: 'Courier New', monospace; font-size: 13px; }
  header { background: var(--panel); padding: 10px 20px; display: flex; align-items: center; gap: 20px; border-bottom: 1px solid var(--border); }
  header h1 { font-size: 16px; font-weight: bold; color: var(--accent); letter-spacing: 2px; }
  #status-bar { margin-left: auto; display: flex; gap: 16px; align-items: center; }
  .badge { padding: 3px 10px; border-radius: 12px; font-size: 11px; font-weight: bold; }
  .badge-manual  { background: #444; }
  .badge-auto    { background: #6a0dad; }
  .badge-parking { background: #c0392b; }
  #hz-label { color: cyan; }
  .main-grid { display: grid; grid-template-columns: 1fr 340px; gap: 10px; padding: 10px; height: calc(100vh - 50px); }
  .left-col { display: flex; flex-direction: column; gap: 10px; }
  .right-col { display: flex; flex-direction: column; gap: 10px; overflow-y: auto; }
  .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 10px; }
  .panel-title { font-size: 11px; font-weight: bold; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
  #cam-feed { width: 100%; border-radius: 4px; background: black; max-height: 380px; object-fit: contain; }
  .telemetry-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; }
  .tele-card { background: #1a1a2e; border-radius: 4px; padding: 8px; text-align: center; }
  .tele-val { font-size: 20px; font-weight: bold; color: var(--accent); }
  .tele-lbl { font-size: 10px; color: #666; margin-top: 2px; }
  .btn { padding: 8px 14px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 12px; font-family: inherit; transition: opacity 0.15s; }
  .btn:hover { opacity: 0.85; }
  .btn-danger  { background: var(--danger); color: white; width: 100%; padding: 12px; font-size: 14px; }
  .btn-accent  { background: var(--accent); color: white; }
  .btn-success { background: var(--success); color: white; }
  .btn-warn    { background: var(--warn); color: black; }
  .btn-gray    { background: #555; color: white; }
  .btn-row { display: flex; gap: 6px; flex-wrap: wrap; }
  .slider-row { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .slider-row label { width: 140px; font-size: 11px; color: #aaa; }
  .slider-row input[type=range] { flex: 1; accent-color: var(--accent); }
  .slider-row span { width: 42px; text-align: right; color: var(--accent); font-weight: bold; font-size: 12px; }
  .indicators { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; }
  .indicator { display: flex; align-items: center; gap: 6px; padding: 4px 6px; border-radius: 4px; background: #1a1a1a; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #333; flex-shrink: 0; }
  .dot.active { background: var(--success); box-shadow: 0 0 6px var(--success); }
  .dot.red    { background: var(--danger); box-shadow: 0 0 6px var(--danger); }
  .dot.orange { background: var(--warn); box-shadow: 0 0 6px var(--warn); }
  #log-box { background: black; border-radius: 4px; height: 180px; overflow-y: auto; padding: 6px; font-size: 11px; line-height: 1.6; }
  .log-INFO     { color: #aaa; }
  .log-SUCCESS  { color: #4caf50; }
  .log-WARN     { color: #ff9800; }
  .log-CRITICAL, .log-DANGER { color: #f44336; }
  .lane-bar-wrap { background: #111; height: 16px; border-radius: 8px; position: relative; overflow: hidden; }
  .lane-bar-center { position: absolute; left: 50%; top: 0; width: 2px; height: 100%; background: #444; }
  .lane-bar-cursor { position: absolute; top: 2px; width: 12px; height: 12px; border-radius: 50%; background: var(--accent); transform: translateX(-50%); transition: left 0.1s; }
  .section-sep { border: none; border-top: 1px solid var(--border); margin: 8px 0; }
</style>
</head>
<body>
<header>
  <h1>&#9881; BFMC 2026</h1>
  <span id="mode-badge" class="badge badge-manual">MANUAL</span>
  <span id="hz-label">0.0 Hz</span>
  <span id="bat-label" style="color:orange">BAT: --%</span>
  <span id="conn-dot" style="color:#f44336">&#9679; STREAM</span>
  <div id="status-bar">
    <span id="rec-label" style="color:#888">&#11044; NOT RECORDING</span>
  </div>
</header>

<div class="main-grid">
  <!-- LEFT COL -->
  <div class="left-col">
    <div class="panel">
      <div class="panel-title">Camera Feed</div>
      <img id="cam-feed" src="/stream" alt="Camera Stream" onerror="this.style.opacity=0.3">
    </div>

    <div class="panel">
      <div class="panel-title">Telemetry</div>
      <div class="telemetry-grid">
        <div class="tele-card"><div class="tele-val" id="t-speed">0</div><div class="tele-lbl">Speed PWM</div></div>
        <div class="tele-card"><div class="tele-val" id="t-steer">0°</div><div class="tele-lbl">Steer</div></div>
        <div class="tele-card"><div class="tele-val" id="t-yaw">0°</div><div class="tele-lbl">Yaw (IMU)</div></div>
        <div class="tele-card"><div class="tele-val" id="t-conf">0.0</div><div class="tele-lbl">Lane Conf</div></div>
        <div class="tele-card"><div class="tele-val" id="t-tx">320</div><div class="tele-lbl">Target X</div></div>
        <div class="tele-card"><div class="tele-val" id="t-lerr">0</div><div class="tele-lbl">Lat Err (px)</div></div>
        <div class="tele-card" style="grid-column:span 2"><div class="tele-val" id="t-anchor" style="font-size:13px">—</div><div class="tele-lbl">Lane Anchor</div></div>
      </div>
      <div style="margin-top:10px">
        <div class="panel-title" style="margin-bottom:4px">Lateral Position</div>
        <div class="lane-bar-wrap">
          <div class="lane-bar-center"></div>
          <div class="lane-bar-cursor" id="lane-cursor" style="left:50%"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- RIGHT COL -->
  <div class="right-col">

    <!-- Emergency Stop -->
    <div class="panel">
      <button class="btn btn-danger" onclick="cmd('e_stop')">&#9940; EMERGENCY STOP</button>
    </div>

    <!-- Mode Controls -->
    <div class="panel">
      <div class="panel-title">Drive Mode</div>
      <div class="btn-row">
        <button class="btn btn-accent"   onclick="cmd('toggle_auto')">Toggle Auto</button>
        <button class="btn btn-warn"     onclick="cmd('toggle_adas')">Toggle ADAS</button>
        <button class="btn btn-gray"     onclick="cmd('clear_route')">Clear Route</button>
      </div>
    </div>

    <!-- Recording -->
    <div class="panel">
      <div class="panel-title">Recording</div>
      <div class="btn-row">
        <button class="btn btn-success" onclick="cmd('start_recording')">&#9210; Start Rec</button>
        <button class="btn btn-danger"  onclick="cmd('stop_recording')">&#9209; Stop Rec</button>
      </div>
    </div>

    <!-- Live Tuning -->
    <div class="panel">
      <div class="panel-title">Live Tuning</div>
      <div class="slider-row">
        <label>Base Speed (PWM)</label>
        <input type="range" id="sl-speed" min="0" max="500" step="5" value="150"
               oninput="document.getElementById('sv-speed').textContent=this.value; cmd('set_base_speed', this.value)">
        <span id="sv-speed">150</span>
      </div>
      <div class="slider-row">
        <label>Steer Multiplier</label>
        <input type="range" id="sl-steer" min="0.1" max="3.0" step="0.1" value="1.0"
               oninput="document.getElementById('sv-steer').textContent=parseFloat(this.value).toFixed(1); cmd('set_steer_mult', this.value)">
        <span id="sv-steer">1.0</span>
      </div>
      <div class="slider-row">
        <label>Sign Detect (m)</label>
        <input type="range" id="sl-sdet" min="1" max="10" step="0.5" value="5.0"
               oninput="document.getElementById('sv-sdet').textContent=parseFloat(this.value).toFixed(1); cmd('set_sign_detect', this.value)">
        <span id="sv-sdet">5.0</span>
      </div>
      <div class="slider-row">
        <label>Sign Act (m)</label>
        <input type="range" id="sl-sact" min="0.5" max="5.0" step="0.1" value="2.0"
               oninput="document.getElementById('sv-sact').textContent=parseFloat(this.value).toFixed(1); cmd('set_sign_act', this.value)">
        <span id="sv-sact">2.0</span>
      </div>
    </div>

    <!-- ADAS Indicators -->
    <div class="panel">
      <div class="panel-title">ADAS Indicators</div>
      <div class="indicators" id="indicators">
        <div class="indicator"><div class="dot" id="ind-stop_sign"></div><span>STOP</span></div>
        <div class="indicator"><div class="dot" id="ind-no_entry"></div><span>NO ENTRY</span></div>
        <div class="indicator"><div class="dot" id="ind-pedestrian"></div><span>PEDESTRIAN</span></div>
        <div class="indicator"><div class="dot" id="ind-red_light"></div><span>RED LIGHT</span></div>
        <div class="indicator"><div class="dot" id="ind-yellow_light"></div><span>YEL LIGHT</span></div>
        <div class="indicator"><div class="dot" id="ind-green_light"></div><span>GRN LIGHT</span></div>
        <div class="indicator"><div class="dot" id="ind-highway"></div><span>HIGHWAY</span></div>
        <div class="indicator"><div class="dot" id="ind-park"></div><span>PARKING</span></div>
        <div class="indicator"><div class="dot" id="ind-overtake"></div><span>OVERTAKE</span></div>
        <div class="indicator"><div class="dot" id="ind-caution"></div><span>CAUTION</span></div>
      </div>
      <hr class="section-sep">
      <div style="font-size:11px; color:#aaa">Active Sign: <span id="t-sign" style="color:var(--warn)">—</span></div>
      <div style="font-size:11px; color:#aaa; margin-top:4px">YOLO: <span id="t-yolo" style="color:#7ec8e3">—</span></div>
    </div>

    <!-- Log Panel -->
    <div class="panel" style="flex:1">
      <div class="panel-title">System Log</div>
      <div id="log-box"></div>
    </div>

  </div>
</div>

<script>
const POLL_MS = 300;

function cmd(action, value) {
  fetch('/api/command', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({action, value: value !== undefined ? value : null})
  });
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

async function pollState() {
  try {
    const r = await fetch('/api/state');
    const s = await r.json();

    document.getElementById('t-speed').textContent = Math.round(s.speed_pwm || 0);
    document.getElementById('t-steer').textContent = (s.steer_deg || 0).toFixed(1) + '°';
    document.getElementById('t-yaw').textContent   = (s.yaw_deg   || 0).toFixed(1) + '°';
    document.getElementById('t-conf').textContent  = (s.lane_confidence || 0).toFixed(2);
    document.getElementById('t-tx').textContent    = Math.round(s.target_x || 320);
    document.getElementById('t-lerr').textContent  = (s.lateral_err_px || 0).toFixed(0);
    document.getElementById('t-anchor').textContent = s.lane_anchor || '—';
    document.getElementById('t-sign').textContent  = s.active_sign || '—';
    document.getElementById('hz-label').textContent = (s.loop_hz || 0).toFixed(1) + ' Hz';
    document.getElementById('bat-label').textContent = 'BAT: ' + (s.battery_pct || 0) + '%';

    const yolo = Array.isArray(s.yolo_labels) ? s.yolo_labels.join(', ') : (s.yolo_labels || '—');
    document.getElementById('t-yolo').textContent = yolo || '—';

    // Lane cursor (target_x maps 150–490 to 0–100%)
    const pct = clamp(((s.target_x || 320) - 150) / (490 - 150) * 100, 0, 100);
    document.getElementById('lane-cursor').style.left = pct + '%';

    // Mode badge
    const badge = document.getElementById('mode-badge');
    const m = (s.mode || 'MANUAL').toUpperCase();
    badge.textContent = m;
    badge.className = 'badge ' + (m.includes('AUTO') ? 'badge-auto' : m.includes('PARK') ? 'badge-parking' : 'badge-manual');

    // Recording indicator
    const recLbl = document.getElementById('rec-label');
    if (s.is_recording) {
      recLbl.style.color = '#f44336';
      recLbl.textContent = '⏺ RECORDING';
    } else {
      recLbl.style.color = '#888';
      recLbl.textContent = '⬜ NOT RECORDING';
    }

    // Sync sliders without triggering oninput (to avoid command echo)
    function syncSlider(id, valId, val, fmt) {
      const el = document.getElementById(id);
      if (document.activeElement !== el) {
        el.value = val;
        document.getElementById(valId).textContent = fmt(val);
      }
    }
    syncSlider('sl-speed', 'sv-speed', s.base_speed || 150, v => Math.round(v));
    syncSlider('sl-steer', 'sv-steer', s.steer_mult || 1.0, v => parseFloat(v).toFixed(1));
    syncSlider('sl-sdet',  'sv-sdet',  s.sign_detect_m || 5.0, v => parseFloat(v).toFixed(1));
    syncSlider('sl-sact',  'sv-sact',  s.sign_act_m   || 2.0, v => parseFloat(v).toFixed(1));

    // ADAS indicators
    const active = s.active_indicators || [];
    document.querySelectorAll('.dot[id^="ind-"]').forEach(d => {
      const key = d.id.replace('ind-', '');
      d.className = 'dot' + (active.includes(key) ? ' active' : '');
    });

  } catch(e) { /* stream may not be ready yet */ }
}

async function pollLog() {
  try {
    const r = await fetch('/api/log');
    const lines = await r.json();
    const box = document.getElementById('log-box');
    box.innerHTML = lines.map(l =>
      `<div class="log-${l.level}">[${l.ts}] ${l.msg}</div>`
    ).join('');
    box.scrollTop = box.scrollHeight;
  } catch(e) {}
}

// Stream img error/reconnect
const camFeed = document.getElementById('cam-feed');
camFeed.onerror = () => {
  setTimeout(() => { camFeed.src = '/stream?' + Date.now(); }, 2000);
};

document.getElementById('conn-dot').style.color = '#4caf50';
setInterval(pollState, POLL_MS);
setInterval(pollLog, 1000);
pollState();
pollLog();
</script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Flask application
# ─────────────────────────────────────────────────────────────────────────────

def _make_app() -> "Flask":
    app = Flask(__name__)
    app.logger.disabled = True
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.ERROR)

    _frame_interval = 1.0 / max(WEB_DASHBOARD_FPS, 1)

    @app.route("/")
    def index():
        return _HTML_TEMPLATE, 200, {"Content-Type": "text/html"}

    @app.route("/stream")
    def stream():
        def generate():
            last_push = 0.0
            while True:
                _frame_event.wait(timeout=1.0)
                _frame_event.clear()
                now = time.monotonic()
                if now - last_push < _frame_interval:
                    continue
                last_push = now
                with _frame_lock:
                    data = _latest_frame
                if not data:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/api/state")
    def api_state():
        with _state_lock:
            snap = dict(_state)
        return jsonify(snap)

    @app.route("/api/log")
    def api_log():
        with _log_lock:
            lines = list(_log_lines)
        return jsonify(lines)

    @app.route("/api/command", methods=["POST"])
    def api_command():
        data = request.get_json(silent=True) or {}
        action = str(data.get("action", "")).strip()
        value  = data.get("value")
        if action:
            with _cmd_lock:
                _pending_commands.append({"action": action, "value": value})
        return jsonify({"ok": True})

    return app


# ─────────────────────────────────────────────────────────────────────────────
#  WebDashboard class
# ─────────────────────────────────────────────────────────────────────────────

class WebDashboard:
    def __init__(self):
        if not _FLASK_AVAILABLE:
            print("[WebDashboard] Flask not installed — web dashboard disabled. pip install flask")
            self._app = None
            return
        self._app = _make_app()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._app is None:
            return
        self._thread = threading.Thread(
            target=self._app.run,
            kwargs={"host": WEB_DASHBOARD_HOST, "port": WEB_DASHBOARD_PORT,
                    "threaded": True, "use_reloader": False, "debug": False},
            daemon=True,
            name="web-dashboard",
        )
        self._thread.start()
        print(f"[WebDashboard] Listening on http://{WEB_DASHBOARD_HOST}:{WEB_DASHBOARD_PORT}")

    # Convenience pass-throughs so main.py only imports WebDashboard
    push_telemetry = staticmethod(push_telemetry)
    push_frame     = staticmethod(push_frame)
    push_log       = staticmethod(push_log)
    pop_commands   = staticmethod(pop_commands)
