"""
Correlate vision / YOLO traffic cues with map-backed path signs for lightweight
global anchoring (no SLAM). One bounded EKF position update per cycle when gated.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _norm(s: str) -> str:
    return str(s).lower().replace("_", "-").strip()


def _label_matches_sign(label: str, sign_type: str) -> bool:
    """Loose match between YOLO class string and signs_database `type` (e.g. stop-sign)."""
    lab = _norm(label)
    st = _norm(sign_type).replace("-sign", "")
    if not st:
        return False
    if st in lab or lab in st:
        return True
    parts = st.split("-")
    return any(p and len(p) > 2 and p in lab for p in parts)


def build_semantic_position_anchor(
    graph: Any,
    path_signs: Optional[List[dict]],
    active_labels: Optional[List[str]],
    sign_approach_m: float,
    est_x: float,
    est_y: float,
    *,
    max_gate_m: float = 38.0,
    max_approach_m: float = 22.0,
) -> Optional[Dict[str, float]]:
    """
    If a non-completed route sign matches current vision and range is plausible, return:

        x, y, var_x, var_y, confidence, sign_type

    Variances are in m^2. Coordinates prefer sign record x,y, else graph node.
    """
    if not path_signs or not active_labels:
        return None
    if sign_approach_m > max_approach_m or sign_approach_m < 0.0:
        return None

    labels = [_norm(x) for x in active_labels if x]

    for ps in path_signs:
        st_stat = str(ps.get("status", ""))
        if "COMPLETED" in st_stat.upper() or "✅" in st_stat:
            continue

        stype = str(ps.get("type", ""))
        if not any(_label_matches_sign(lb, stype) for lb in labels):
            continue

        nid = str(ps.get("node", ""))
        if "x" in ps and "y" in ps:
            ax = float(ps["x"])
            ay = float(ps["y"])
        elif nid in graph.nodes:
            nd = graph.nodes[nid]
            ax = float(nd.get("x", 0.0))
            ay = float(nd.get("y", 0.0))
        else:
            continue

        dist = math.hypot(est_x - ax, est_y - ay)
        if dist > max_gate_m:
            continue

        base_var = 2.8 + 0.09 * (sign_approach_m ** 2)
        gate = max(0.2, 1.0 - dist / max(max_gate_m, 1e-6))
        conf = float(min(1.0, gate * (1.15 / (0.4 + sign_approach_m))))
        var = max(0.45, base_var * (1.35 - 0.3 * conf))

        return {
            "x": ax,
            "y": ay,
            "var_x": var,
            "var_y": var,
            "confidence": conf,
            "sign_type": stype,
        }

    return None
