"""
Sensor-driven localization (EKF + map matching + lane vision).

Per cycle: predict with bicycle kinematics blended with IMU yaw-rate; correct with
map polyline (mission path or nearest graph edge) and lane metric observations.

State lives in **GraphML world meters** (same frame as MapEngine), not path-interpolated pose.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from localization.ekf_vehicle import VehicleEKF4, wrap_pi
from localization.graph_map_matcher import nearest_edge_projection
from localization.route_geometry import RouteGeometry
from localization.types import LocalizationDebug


class FusedLocalizer:
    def __init__(self, graph, wheelbase_m: float = 0.23):
        self._graph = graph
        self.wheelbase_m = float(wheelbase_m)
        self.ekf = VehicleEKF4(wheelbase_m, w_bicycle=0.55)

        self._geom: Optional[RouteGeometry] = None
        self._path_key: Optional[Tuple[str, ...]] = None

        self._last_imu_yaw_deg: Optional[float] = None
        self._lost_frames = 0
        self._debug = LocalizationDebug()

    @property
    def s(self) -> float:
        """Along-route arc length (m) when a mission path is used; else 0."""
        sm = self._debug.path_s_m
        return float(max(0.0, sm)) if sm >= 0.0 else 0.0

    @property
    def route_total_m(self) -> float:
        if self._geom is not None and self._geom.is_valid():
            return float(self._geom.total_length)
        return 0.0

    @property
    def last_debug(self) -> LocalizationDebug:
        return self._debug

    def invalidate_geometry(self) -> None:
        self._geom = None
        self._path_key = None

    def _ensure_geometry(self, path: List) -> bool:
        key = tuple(str(n) for n in path)
        if self._path_key != key or self._geom is None:
            self._geom = RouteGeometry(self._graph, path)
            self._path_key = key
        return self._geom is not None and self._geom.is_valid()

    def curvature_at_s_for_path(self, path: List, s: float) -> float:
        if not path or len(path) < 2:
            return 0.0
        if not self._ensure_geometry(path):
            return 0.0
        return self._geom.curvature_at_s(s)

    def resync_xy(self, path: List, x: float, y: float) -> None:
        """Map-click re-anchor (DRIVE mode)."""
        if path and len(path) > 1 and self._ensure_geometry(path):
            geom = self._geom
            assert geom is not None
            s0, _ = geom.project_xy_to_s_e(x, y)
            _, _, psi0 = geom.point_tangent_at_s(s0)
            v0 = max(0.0, self.ekf.state()[3]) if self.ekf.initialized else 0.0
            self.ekf.set_state_covariance(x, y, psi0, v0, (0.07, 0.07, 0.1, 0.4))
        else:
            v0 = max(0.0, self.ekf.state()[3]) if self.ekf.initialized else 0.0
            self.ekf.set_state_covariance(x, y, 0.0, v0, (0.14, 0.14, 0.4, 0.5))

    def sync_s_after_teleport(self, path: List, teleport_s: float) -> None:
        """Backward-compatible hook: snap pose to route location after sign completion."""
        if not path or len(path) < 2 or not self._ensure_geometry(path):
            return
        geom = self._geom
        assert geom is not None
        s_clamped = float(np.clip(teleport_s, 0.0, geom.total_length))
        x0, y0, psi0 = geom.point_tangent_at_s(s_clamped)
        v0 = max(0.0, self.ekf.state()[3]) if self.ekf.initialized else 0.0
        self.ekf.set_state_covariance(x0, y0, psi0, v0, (0.025, 0.025, 0.06, 0.12))

    def sync_pose_to_node(self, x: float, y: float, psi_rad: float, v_ms: float = 0.0) -> None:
        """Hard reset at a graph node (meters, rad)."""
        self.ekf.set_state_covariance(x, y, psi_rad, max(0.0, v_ms), (0.02, 0.02, 0.04, 0.12))

    def apply_event_anchor(
        self,
        x: float,
        y: float,
        psi_rad: float,
        std_xy: Tuple[float, float] = (0.55, 0.55),
        std_psi: float = 0.18,
    ) -> None:
        """
        Discrete global anchor (e.g. sign completion) — replaces teleport snapping with
        an explicit covariance reset at the map node / tangent.
        """
        v0 = max(0.0, self.ekf.state()[3]) if self.ekf.initialized else 0.0
        self.ekf.set_state_covariance(
            x, y, wrap_pi(psi_rad), v0, (std_xy[0], std_xy[1], std_psi, 0.28)
        )

    def update(
        self,
        dt: float,
        path: Optional[List],
        v_cmd_ms: float,
        steer_deg: float,
        lane_result: Any,
        imu_yaw_deg: float,
        imu_ok: bool,
        *,
        path_just_changed: bool,
        car_x_hint: float,
        car_y_hint: float,
        semantic_anchor: Optional[dict] = None,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Predict + correct. Returns (x, y, yaw_rad, kappa_path) or None if not initialized
        and hints are invalid (extremely rare).
        """
        use_path = path is not None and len(path) > 1 and self._ensure_geometry(path)

        # --- Re-anchor when mission path changes (before dynamics prediction) ---
        if path_just_changed and use_path:
            geom = self._geom
            assert geom is not None
            s0, _ = geom.project_xy_to_s_e(car_x_hint, car_y_hint)
            _, _, psi0 = geom.point_tangent_at_s(s0)
            v0 = max(0.0, self.ekf.state()[3]) if self.ekf.initialized else max(0.0, v_cmd_ms)
            self.ekf.set_state_covariance(car_x_hint, car_y_hint, psi0, v0, (0.07, 0.07, 0.1, 0.35))

        # --- IMU yaw rate ---
        yaw_rate_imu = 0.0
        if self._last_imu_yaw_deg is not None:
            yaw_rate_imu = math.radians(imu_yaw_deg - self._last_imu_yaw_deg) / max(dt, 1e-6)
            yaw_rate_imu = float(np.clip(yaw_rate_imu, -2.5, 2.5))
        self._last_imu_yaw_deg = float(imu_yaw_deg)

        # --- Bootstrap ---
        if not self.ekf.initialized:
            psi0 = math.radians(imu_yaw_deg)
            if use_path:
                geom = self._geom
                assert geom is not None
                s0, _ = geom.project_xy_to_s_e(car_x_hint, car_y_hint)
                x0, y0, psi0 = geom.point_tangent_at_s(s0)
                self.ekf.initialize(x0, y0, psi0, max(0.0, v_cmd_ms))
            else:
                self.ekf.initialize(car_x_hint, car_y_hint, psi0, max(0.0, v_cmd_ms))

        # --- Predict ---
        self.ekf.predict(dt, steer_deg, max(0.0, v_cmd_ms), yaw_rate_imu if imu_ok else 0.0)
        xp, yp, psip, vp = self.ekf.state()

        lane_conf = 0.0
        e_lane_m = 0.0
        heading_lane = 0.0
        signed_curv = 0.0
        if lane_result is not None:
            lane_conf = float(np.clip(getattr(lane_result, "confidence", 0.0), 0.0, 1.0))
            lw = max(float(getattr(lane_result, "lane_width_px", 280.0)), 50.0)
            ppm = lw / 0.35
            e_lane_m = float(getattr(lane_result, "lateral_error_px", 0.0)) / max(ppm, 1e-6)
            heading_lane = float(getattr(lane_result, "heading_rad", 0.0))
            signed_curv = float(getattr(lane_result, "signed_curvature", 0.0))

        kappa_out = 0.0
        s_m = -1.0
        seg_idx = -1
        e_ct = 0.0
        map_conf = 0.35

        if use_path:
            geom = self._geom
            assert geom is not None
            xk, yk, _, _ = self.ekf.state()
            s_m, e_ct = geom.project_xy_to_s_e(xk, yk)
            x0, y0, psi_ref = geom.point_tangent_at_s(s_m)
            kappa_out = geom.curvature_at_s(s_m)
            seg_idx = geom.segment_index_at_s(s_m)
            nx, ny = -math.sin(psi_ref), math.cos(psi_ref)

            r_map_e = float(np.clip(0.55 + 2.5 * abs(kappa_out) + (2.0 if lane_conf < 0.25 else 0.0), 0.35, 6.0))
            self.ekf.correct_lateral_linear(nx, ny, x0, y0, 0.0, r_map_e)

            r_map_psi = float(np.clip(0.18 + 0.6 * (1.0 - min(1.0, lane_conf)), 0.08, 0.55))
            self.ekf.correct_heading(psi_ref, r_map_psi)

            map_conf = float(np.clip(1.0 / (1.0 + 0.35 * r_map_e), 0.15, 0.95))

            if lane_conf > 0.18:
                curve_f = 1.0 + min(3.8, 70.0 * abs(signed_curv))
                r_lane_e = max(0.12, 0.55 / (0.25 + lane_conf)) * curve_f
                self.ekf.correct_lateral_linear(nx, ny, x0, y0, e_lane_m, r_lane_e)
                z_psi_lane = wrap_pi(psi_ref + heading_lane)
                r_lane_psi = max(0.06, 0.22 / (0.2 + lane_conf)) * (1.0 + 0.28 * min(2.2, 45.0 * abs(signed_curv)))
                self.ekf.correct_heading(z_psi_lane, r_lane_psi)
        else:
            hit = nearest_edge_projection(self._graph, *self.ekf.state()[:2])
            if hit is not None:
                qx, qy, nx, ny, psi_ref, d2 = hit
                dist = math.sqrt(max(d2, 0.0))
                map_conf = float(np.clip(1.2 / (1.0 + dist), 0.1, 0.7))
                r_edge = float(np.clip(0.9 + 0.02 * dist, 0.5, 4.0))
                self.ekf.correct_lateral_linear(nx, ny, qx, qy, 0.0, r_edge)
                self.ekf.correct_heading(psi_ref, float(np.clip(0.35 + 0.04 * dist, 0.15, 0.8)))

        imu_conf = 0.85 if imu_ok else 0.2

        conf_sem = 0.0
        tag_sem = ""
        if semantic_anchor and float(semantic_anchor.get("confidence", 0.0)) > 0.17:
            self.ekf.correct_position_xy(
                float(semantic_anchor["x"]),
                float(semantic_anchor["y"]),
                float(semantic_anchor["var_x"]),
                float(semantic_anchor["var_y"]),
            )
            conf_sem = float(np.clip(semantic_anchor.get("confidence", 0.0), 0.0, 1.0))
            tag_sem = str(semantic_anchor.get("sign_type", ""))

        xf, yf, psif, vf = self.ekf.state()
        Pxy = self.ekf.horizontal_covariance()
        drift = float(math.sqrt(max(0.5 * (Pxy[0, 0] + Pxy[1, 1]), 0.0)))

        if lane_conf < 0.12:
            self._lost_frames += 1
        else:
            self._lost_frames = max(0, self._lost_frames - 2)
        lost = self._lost_frames > 28 or drift > 2.5

        tot_conf = float(
            np.clip(
                0.4 * lane_conf + 0.3 * map_conf + 0.18 * imu_conf + 0.12 * conf_sem,
                0.0,
                1.0,
            )
        )
        tot_conf *= float(np.clip(math.exp(-0.45 * drift), 0.35, 1.0))

        self._debug = LocalizationDebug(
            x=xf,
            y=yf,
            yaw_rad=psif,
            v_ms=vf,
            x_pred=xp,
            y_pred=yp,
            yaw_pred_rad=psip,
            v_pred_ms=vp,
            path_s_m=s_m,
            path_segment_idx=seg_idx,
            cross_track_m=e_ct,
            confidence_total=tot_conf,
            confidence_lane=lane_conf,
            confidence_map=map_conf,
            confidence_imu=imu_conf,
            confidence_semantic=conf_sem,
            semantic_tag=tag_sem,
            drift_pos_m=drift,
            lost=lost,
            P_xy=((float(Pxy[0, 0]), float(Pxy[0, 1])), (float(Pxy[1, 0]), float(Pxy[1, 1]))),
        )

        return xf, yf, psif, kappa_out
