import cv2
import numpy as np
from config import (
    LANE_BLUR_KERNEL, LANE_THRESHOLD,
    LANE_BRIGHT_LOW, LANE_BRIGHT_HIGH, LANE_MORPH_KERNEL,
)


def preprocess_image(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Extract white lane markings from a bird's-eye-view BGR frame.

    Pipeline:
      1. Convert to LAB, take L channel (raw — no CLAHE, which amplifies road noise)
      2. Gently normalise brightness for indoor lighting variation
      3. Gaussian blur to suppress residual texture
      4. Global threshold: white lane lines (L≈200+) vs dark carpet (L≈80-130)
      5. Morphological close to bridge dashed-line gaps
    """
    lab   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L     = lab[:, :, 0]

    mean_l = np.mean(L)
    if mean_l < LANE_BRIGHT_LOW:
        L = cv2.convertScaleAbs(L, alpha=1.0 + (LANE_BRIGHT_LOW  - mean_l) / 100, beta=0)
    elif mean_l > LANE_BRIGHT_HIGH:
        L = cv2.convertScaleAbs(L, alpha=1.0 - (mean_l - LANE_BRIGHT_HIGH) / 250, beta=0)

    L_blur = cv2.GaussianBlur(L, LANE_BLUR_KERNEL, 0)
    _, binary = cv2.threshold(L_blur, LANE_THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, LANE_MORPH_KERNEL)
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
