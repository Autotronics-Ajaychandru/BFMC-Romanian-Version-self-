import tkinter as tk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from config import THEME
except ImportError:
    THEME = {"bg": "#1e1e1e", "panel": "#252526", "accent": "#007acc", "success": "#4caf50"}


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _box_edges(lo, hi):
    """Return 12 edge (p0, p1) pairs for an axis-aligned box."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    c = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ])
    idx = [(0,1),(1,2),(2,3),(3,0), (4,5),(5,6),(6,7),(7,4), (0,4),(1,5),(2,6),(3,7)]
    return [(c[a], c[b]) for a, b in idx]


def _rotation_zyx(yaw_deg, pitch_deg, roll_deg):
    """ZYX intrinsic rotation matrix: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    y, p, r = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [0,          0,         1]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [ 0,         1, 0        ],
                   [-np.sin(p), 0, np.cos(p)]])
    Rx = np.array([[1, 0,         0        ],
                   [0, np.cos(r), -np.sin(r)],
                   [0, np.sin(r),  np.cos(r)]])
    return Rz @ Ry @ Rx


# Car geometry constants (in local car frame: X=forward, Y=left, Z=up)
_BODY_LO  = (-1.05, -0.45, -0.28)
_BODY_HI  = ( 1.05,  0.45,  0.28)
_CABIN_LO = (-0.30, -0.38,  0.28)
_CABIN_HI = ( 0.60,  0.38,  0.58)
_WHEEL_CENTERS = [(-0.75, -0.52, -0.28), (-0.75, 0.52, -0.28),
                  ( 0.75, -0.52, -0.28), ( 0.75, 0.52, -0.28)]
_WHEEL_R  = 0.22
_WHEEL_TH = np.linspace(0, 2 * np.pi, 24)


class IMU3DPanel:
    """
    Pop-up Toplevel that shows a live-rotating 3D car model driven by
    the IMU yaw / pitch / roll.  Call open() from the main thread only.
    """

    def __init__(self, parent: tk.Misc, imu_sensor):
        self._imu     = imu_sensor
        self._cal_yaw = self._cal_pitch = self._cal_roll = 0.0
        self._alive   = True

        # ── Window ────────────────────────────────────────────
        self.win = tk.Toplevel(parent)
        self.win.title("IMU 3D Orientation")
        self.win.geometry("500x560")
        self.win.minsize(420, 480)
        self.win.configure(bg=THEME["bg"])
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Matplotlib figure (dark) ───────────────────────────
        self._fig = Figure(figsize=(4.8, 4.0), dpi=95, facecolor=THEME["bg"])
        self._ax  = self._fig.add_subplot(111, projection="3d")
        self._ax.set_facecolor(THEME["bg"])
        self._fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.0)

        self._canvas = FigureCanvasTkAgg(self._fig, master=self.win)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── Numeric readout ────────────────────────────────────
        ro = tk.Frame(self.win, bg=THEME["panel"], pady=4)
        ro.pack(fill=tk.X, padx=8)
        _font = ("Courier", 12, "bold")
        self._lbl_yaw   = tk.Label(ro, text="YAW:    +0.0°", bg=THEME["panel"], fg="#00e5ff", font=_font, width=15, anchor="w")
        self._lbl_pitch = tk.Label(ro, text="PITCH:  +0.0°", bg=THEME["panel"], fg="#76ff03", font=_font, width=15, anchor="w")
        self._lbl_roll  = tk.Label(ro, text="ROLL:   +0.0°", bg=THEME["panel"], fg="#ff9100", font=_font, width=15, anchor="w")
        for lbl in (self._lbl_yaw, self._lbl_pitch, self._lbl_roll):
            lbl.pack(side=tk.LEFT, expand=True)

        # ── Calibrate button ───────────────────────────────────
        btn_row = tk.Frame(self.win, bg=THEME["bg"])
        btn_row.pack(fill=tk.X, padx=8, pady=(4, 8))
        self._btn_cal = tk.Button(
            btn_row, text="CALIBRATE  (Set current pose as 0, 0, 0)",
            bg=THEME["accent"], fg="white", relief="flat",
            font=("Helvetica", 10, "bold"), activebackground="#005fa3",
            command=self._calibrate,
        )
        self._btn_cal.pack(fill=tk.X)

        self._schedule_update()

    # ── Public ────────────────────────────────────────────────
    def lift(self):
        self.win.lift()
        self.win.focus_force()

    # ── Private helpers ───────────────────────────────────────
    def _calibrate(self):
        self._cal_yaw   = self._imu.get_yaw()
        self._cal_pitch = self._imu.get_pitch()
        self._cal_roll  = self._imu.get_roll()
        self._btn_cal.config(text="CALIBRATED  ✓  (origin reset)", bg=THEME["success"])
        self.win.after(1800, lambda: self._btn_cal.config(
            text="CALIBRATE  (Set current pose as 0, 0, 0)", bg=THEME["accent"]
        ))

    def _on_close(self):
        self._alive = False
        self.win.destroy()

    def _schedule_update(self):
        if not self._alive:
            return
        try:
            self._render()
        except Exception:
            pass
        self.win.after(50, self._schedule_update)   # 20 Hz

    def _render(self):
        yaw   = self._imu.get_yaw()   - self._cal_yaw
        pitch = self._imu.get_pitch() - self._cal_pitch
        roll  = self._imu.get_roll()  - self._cal_roll

        self._lbl_yaw.config(  text=f"YAW:  {yaw:+7.1f}°")
        self._lbl_pitch.config(text=f"PITCH:{pitch:+7.1f}°")
        self._lbl_roll.config( text=f"ROLL: {roll:+7.1f}°")

        ax = self._ax
        ax.cla()

        # Axes styling (must redo after cla)
        ax.set_facecolor(THEME["bg"])
        ax.set_xlim(-1.8, 1.8); ax.set_ylim(-1.8, 1.8); ax.set_zlim(-1.8, 1.8)
        ax.set_xlabel("X", color="#888", fontsize=8, labelpad=2)
        ax.set_ylabel("Y", color="#888", fontsize=8, labelpad=2)
        ax.set_zlabel("Z", color="#888", fontsize=8, labelpad=2)
        ax.tick_params(colors="#666", labelsize=6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#333333")
        for spine in (ax.xaxis.line, ax.yaxis.line, ax.zaxis.line):
            spine.set_color("#555555")
        ax.set_title(
            f"YAW {yaw:+.1f}°   PITCH {pitch:+.1f}°   ROLL {roll:+.1f}°",
            color="white", fontsize=9, pad=6,
        )

        R = _rotation_zyx(yaw, pitch, roll)

        def _plot_edges(edges, color, lw):
            for p0, p1 in edges:
                rp0, rp1 = R @ p0, R @ p1
                ax.plot([rp0[0], rp1[0]], [rp0[1], rp1[1]], [rp0[2], rp1[2]],
                        color=color, linewidth=lw)

        # Body + cabin wireframes
        _plot_edges(_box_edges(_BODY_LO,  _BODY_HI),  "#4fc3f7", 1.6)
        _plot_edges(_box_edges(_CABIN_LO, _CABIN_HI), "#b0bec5", 1.2)

        # Wheels — circles in the local XZ plane at each corner
        for cx, cy, cz in _WHEEL_CENTERS:
            wx = cx + _WHEEL_R * np.sin(_WHEEL_TH)
            wy = np.full_like(wx, cy)
            wz = cz + _WHEEL_R * np.cos(_WHEEL_TH)
            pts = (R @ np.vstack([wx, wy, wz]))
            ax.plot(pts[0], pts[1], pts[2], color="#546e7a", linewidth=1.2)

        # Reference axes arrows (show vehicle frame)
        for vec, col, lbl in [([1.4, 0, 0], "#ef5350", "X"),
                               ([0, 1.4, 0], "#66bb6a", "Y"),
                               ([0, 0, 1.4], "#42a5f5", "Z")]:
            rv = R @ np.array(vec, dtype=float)
            ax.quiver(0, 0, 0, rv[0], rv[1], rv[2],
                      color=col, arrow_length_ratio=0.18, linewidth=1.8)
            ax.text(rv[0] * 1.12, rv[1] * 1.12, rv[2] * 1.12, lbl,
                    color=col, fontsize=9, fontweight="bold")

        self._canvas.draw_idle()
