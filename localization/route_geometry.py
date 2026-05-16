"""
Polyline geometry for the active navigation path (GraphML node coordinates in meters).
Used for arc-length parameterization, tangents, curvature, and XY↔closest-point projection.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


class RouteGeometry:
    """Piecewise-linear centerline from an ordered list of graph node ids."""

    def __init__(self, graph, path_nodes: List):
        self._graph = graph
        self.path_nodes = [str(n) for n in path_nodes]
        self.points: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self.cumlen: np.ndarray = np.zeros(0, dtype=np.float64)
        self.seg_len: np.ndarray = np.zeros(0, dtype=np.float64)
        self.total_length: float = 0.0
        self._build()

    def _build(self) -> None:
        pts = []
        G = self._graph
        for nid in self.path_nodes:
            if nid not in G.nodes:
                continue
            d = G.nodes[nid]
            pts.append((float(d.get("x", 0.0)), float(d.get("y", 0.0))))
        if len(pts) < 2:
            self.points = np.zeros((0, 2), dtype=np.float64)
            self.cumlen = np.zeros(0, dtype=np.float64)
            self.seg_len = np.zeros(0, dtype=np.float64)
            self.total_length = 0.0
            return
        self.points = np.array(pts, dtype=np.float64)
        dif = np.diff(self.points, axis=0)
        self.seg_len = np.hypot(dif[:, 0], dif[:, 1])
        self.cumlen = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total_length = float(self.cumlen[-1])

    def is_valid(self) -> bool:
        return self.points.shape[0] >= 2 and self.total_length > 1e-6

    def _clamp_s(self, s: float) -> float:
        return float(np.clip(s, 0.0, max(self.total_length, 0.0)))

    def point_tangent_at_s(self, s: float) -> Tuple[float, float, float]:
        """Return (x, y, psi_rad) on the polyline at arc length s (tangent = forward direction)."""
        s = self._clamp_s(s)
        if not self.is_valid():
            return 0.0, 0.0, 0.0
        # find segment
        idx = int(np.searchsorted(self.cumlen, s, side="right") - 1)
        idx = max(0, min(idx, len(self.seg_len) - 1))
        seg_start = self.cumlen[idx]
        seg_l = max(self.seg_len[idx], 1e-9)
        t = (s - seg_start) / seg_l
        p0 = self.points[idx]
        p1 = self.points[idx + 1]
        x = float(p0[0] + t * (p1[0] - p0[0]))
        y = float(p0[1] + t * (p1[1] - p0[1]))
        psi = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        return x, y, psi

    def curvature_at_s(self, s: float) -> float:
        """
        Discrete geometric curvature κ ≈ Δψ / Δs using adjacent segment headings.
        Positive = left turn in standard math (CCW positive angle).
        """
        s = self._clamp_s(s)
        if not self.is_valid() or len(self.seg_len) < 1:
            return 0.0
        ds = 0.5
        _, _, psi_m = self.point_tangent_at_s(max(0.0, s - ds))
        _, _, psi_p = self.point_tangent_at_s(min(self.total_length, s + ds))
        dpsi = math.atan2(math.sin(psi_p - psi_m), math.cos(psi_p - psi_m))
        span = min(2.0 * ds, max(self.total_length, 1e-6))
        return dpsi / span

    def project_xy_to_s_e(self, x: float, y: float) -> Tuple[float, float]:
        """
        Orthogonal projection of (x,y) onto the polyline.
        Returns (s, e) where e is signed lateral offset (m): positive = point lies to the
        left of the forward tangent (left normal: (-sin ψ, cos ψ)).
        """
        if not self.is_valid():
            return 0.0, 0.0
        best_d2 = float("inf")
        best_s = 0.0
        best_e = 0.0
        px, py = float(x), float(y)
        for i in range(len(self.seg_len)):
            p0 = self.points[i]
            p1 = self.points[i + 1]
            ax, ay = p0[0], p0[1]
            bx, by = p1[0], p1[1]
            abx, aby = bx - ax, by - ay
            seg_l2 = abx * abx + aby * aby
            if seg_l2 < 1e-12:
                continue
            t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / seg_l2))
            qx = ax + t * abx
            qy = ay + t * aby
            dx, dy = px - qx, py - qy
            d2 = dx * dx + dy * dy
            psi = math.atan2(aby, abx)
            nx, ny = -math.sin(psi), math.cos(psi)  # left normal
            e = dx * nx + dy * ny
            s_here = float(self.cumlen[i] + t * self.seg_len[i])
            if d2 < best_d2:
                best_d2 = d2
                best_s = s_here
                best_e = e
        return best_s, best_e

    def segment_index_at_s(self, s: float) -> int:
        """Index i such that segment (points[i], points[i+1]) contains arc length s."""
        s = self._clamp_s(s)
        if not self.is_valid():
            return 0
        idx = int(np.searchsorted(self.cumlen, s, side="right") - 1)
        return int(max(0, min(idx, len(self.seg_len) - 1)))
