"""Localization debug / telemetry for GUI and logging."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class LocalizationDebug:
    """Single-cycle localization diagnostics (sensor-driven sim / EKF)."""

    x: float = 0.0
    y: float = 0.0
    yaw_rad: float = 0.0
    v_ms: float = 0.0
    # Pose after prediction, before measurement updates
    x_pred: float = 0.0
    y_pred: float = 0.0
    yaw_pred_rad: float = 0.0
    v_pred_ms: float = 0.0
    # Along-route arc length (m) for mission path; -1 if N/A
    path_s_m: float = -1.0
    path_segment_idx: int = -1
    cross_track_m: float = 0.0
    # Confidence in [0, 1]
    confidence_total: float = 0.0
    confidence_lane: float = 0.0
    confidence_map: float = 0.0
    confidence_imu: float = 0.0
    confidence_semantic: float = 0.0
    semantic_tag: str = ""
    # Drift proxy: position std (m) from EKF horizontal slice
    drift_pos_m: float = 0.0
    lost: bool = False
    # 2x2 horizontal covariance (m^2) for ellipse drawing
    P_xy: Tuple[Tuple[float, float], Tuple[float, float]] = field(
        default_factory=lambda: ((0.04, 0.0), (0.0, 0.04))
    )
