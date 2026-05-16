import cv2
import numpy as np
from config import LANE_SRC_PTS, LANE_DST_PTS


class PerspectiveTransformer:
    def __init__(self):
        src = np.float32(LANE_SRC_PTS)
        dst = np.float32(LANE_DST_PTS)
        self.M_forward = cv2.getPerspectiveTransform(src, dst)
        self.M_inv     = cv2.getPerspectiveTransform(dst, src)

    def warp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.M_forward, (640, 480))

    def unwarp(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.M_inv, (640, 480))
