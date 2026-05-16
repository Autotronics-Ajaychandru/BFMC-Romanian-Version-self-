"""
Nearest drivable segment on the full GraphML map (world meters).
Used when no mission path is active — weak map matching / association.
"""
from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple


def nearest_edge_projection(
    graph: Any, x: float, y: float
) -> Optional[Tuple[float, float, float, float, float, float]]:
    """
    Returns (x0, y0, nx, ny, psi_tangent, dist2) for the closest point on any graph edge,
    where (nx,ny) is the left unit normal and (x0,y0) is the foot on the segment.
    """
    best: Optional[Tuple[float, float, float, float, float, float]] = None
    px, py = float(x), float(y)

    for u, v in graph.edges():
        su, sv = str(u), str(v)
        if su not in graph.nodes or sv not in graph.nodes:
            continue
        du = graph.nodes[su]
        dv = graph.nodes[sv]
        ax, ay = float(du.get("x", 0)), float(du.get("y", 0))
        bx, by = float(dv.get("x", 0)), float(dv.get("y", 0))
        abx, aby = bx - ax, by - ay
        seg_l2 = abx * abx + aby * aby
        if seg_l2 < 1e-14:
            continue
        t = max(0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / seg_l2))
        qx = ax + t * abx
        qy = ay + t * aby
        dx, dy = px - qx, py - qy
        d2 = dx * dx + dy * dy
        psi = math.atan2(aby, abx)
        nx, ny = -math.sin(psi), math.cos(psi)
        cand = (qx, qy, nx, ny, psi, d2)
        if best is None or d2 < best[-1]:
            best = cand

    if best is None:
        return None
    return best
