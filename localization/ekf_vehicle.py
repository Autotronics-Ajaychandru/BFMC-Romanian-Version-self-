"""
4-state vehicle EKF: [x, y, psi, v] in map meters / rad / m/s.

Prediction: kinematic bicycle yaw rate blended with IMU yaw-rate;
             velocity toward commanded speed (first-order lag).
Corrections: scalar Joseph updates with linearized H rows.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def wrap_pi(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


class VehicleEKF4:
    def __init__(self, wheelbase_m: float, w_bicycle: float = 0.55):
        self.L = max(float(wheelbase_m), 0.05)
        self.w_bicycle = float(w_bicycle)
        self.w_imu = float(1.0 - w_bicycle)
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 0.25
        self._initialized = False

    def initialize(self, x0: float, y0: float, psi0: float, v0: float) -> None:
        self.x[:] = (x0, y0, wrap_pi(psi0), max(0.0, v0))
        self.P = np.diag([0.08, 0.08, 0.06, 0.25]).astype(np.float64)
        self._initialized = True

    @property
    def initialized(self) -> bool:
        return self._initialized

    def set_state_covariance(
        self, x0: float, y0: float, psi0: float, v0: float, diag_std: Tuple[float, float, float, float]
    ) -> None:
        self.x[:] = (x0, y0, wrap_pi(psi0), max(0.0, v0))
        d = np.array(diag_std, dtype=np.float64) ** 2
        self.P = np.diag(d)
        self._initialized = True

    def state(self) -> Tuple[float, float, float, float]:
        return float(self.x[0]), float(self.x[1]), float(self.x[2]), float(self.x[3])

    def predict(self, dt: float, steer_deg: float, v_cmd_ms: float, yaw_rate_imu_rad_s: float) -> None:
        if not self._initialized or dt <= 0.0:
            return
        L = self.L
        steer = float(np.clip(np.radians(steer_deg), np.radians(-28.0), np.radians(28.0)))
        x, y, psi, v = self.x
        yd_bike = (v / L) * math.tan(steer)
        yd = self.w_bicycle * yd_bike + self.w_imu * float(yaw_rate_imu_rad_s)

        tau_v = 0.22
        alpha_v = min(1.0, dt / tau_v) if tau_v > 1e-6 else 1.0
        v_new = v + (max(0.0, v_cmd_ms) - v) * alpha_v

        x_new = x + v * math.cos(psi) * dt
        y_new = y + v * math.sin(psi) * dt
        psi_new = wrap_pi(psi + yd * dt)

        # Jacobian F = d f / d x
        F = np.eye(4, dtype=np.float64)
        F[0, 2] = -v * math.sin(psi) * dt
        F[1, 2] = v * math.cos(psi) * dt
        F[0, 3] = math.cos(psi) * dt
        F[1, 3] = math.sin(psi) * dt
        F[2, 3] = (math.tan(steer) / L) * self.w_bicycle * dt

        qxy = 0.015 * dt + 0.004 * abs(v) * dt
        qpsi = (0.06 * dt) ** 2 + (0.04 * abs(yd) * dt) ** 2
        qv = (0.35 * dt) ** 2
        Q = np.diag([qxy, qxy, qpsi, qv]).astype(np.float64)

        self.x[:] = (x_new, y_new, psi_new, v_new)
        self.P = F @ self.P @ F.T + Q
        self.x[2] = wrap_pi(self.x[2])

    def _joseph_scalar(self, H: np.ndarray, innovation: float, R: float) -> None:
        """H shape (1,4), innovation scalar, R scalar measurement variance."""
        H = H.reshape(1, 4)
        S = float(H @ self.P @ H.T + R)
        if S < 1e-9:
            return
        K = (self.P @ H.T / S).reshape(4)
        self.x = self.x + K * innovation
        self.x[2] = wrap_pi(self.x[2])
        I_KH = np.eye(4) - np.outer(K, H)
        self.P = I_KH @ self.P @ I_KH.T + R * np.outer(K, K)

    def correct_heading(self, z_psi: float, R_psi: float) -> None:
        """Observe absolute heading z_psi (rad)."""
        if not self._initialized:
            return
        H = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float64)
        y = wrap_pi(z_psi - float(self.x[2]))
        self._joseph_scalar(H, y, R_psi)

    def correct_position_xy(self, zx: float, zy: float, var_x: float, var_y: float) -> None:
        """Observe global x and y (Joseph scalar updates). var_* are measurement variances (m^2)."""
        if not self._initialized:
            return
        vx = max(float(var_x), 0.08)
        vy = max(float(var_y), 0.08)
        Hx = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
        self._joseph_scalar(Hx, float(zx - self.x[0]), vx)
        Hy = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        self._joseph_scalar(Hy, float(zy - self.x[1]), vy)

    def correct_lateral_linear(self, nx: float, ny: float, x0: float, y0: float, z_e: float, R_e: float) -> None:
        """
        Observe lateral offset z_e ≈ nx*(x-x0) + ny*(y-y0) with fixed foot (x0,y0) and normal (nx,ny).
        """
        if not self._initialized:
            return
        H = np.array([[nx, ny, 0.0, 0.0]], dtype=np.float64)
        h = nx * (self.x[0] - x0) + ny * (self.x[1] - y0)
        y = float(z_e - h)
        self._joseph_scalar(H, y, R_e)

    def horizontal_covariance(self) -> np.ndarray:
        return np.array(
            [[self.P[0, 0], self.P[0, 1]], [self.P[1, 0], self.P[1, 1]]], dtype=np.float64
        )
