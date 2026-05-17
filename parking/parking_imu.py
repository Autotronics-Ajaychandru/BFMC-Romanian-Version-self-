import math
from .parking_config import IMU_DEADBAND, VELOCITY_DAMPING

class ParkingDistanceTracker:
    def __init__(self):
        self.distance_cm = 0.0
        self.velocity = 0.0
        self.acc_offset = 0.0

    def reset(self):
        self.distance_cm = 0.0
        self.velocity = 0.0
        self.acc_offset = 0.0

    def update(self, accel_forward, dt):
        """
        Integrate forward acceleration (m/s^2) to estimate distance in cm.
        """
        accel = accel_forward - self.acc_offset
        if abs(accel) < IMU_DEADBAND:
            accel = 0.0
        self.velocity += accel * dt
        self.velocity *= VELOCITY_DAMPING
        dist_m = self.velocity * dt
        self.distance_cm += dist_m * 100.0
