# Lane Detector Module - Comprehensive Technical Documentation

## Table of Contents
1. [Module Overview](#module-overview)
2. [Dependencies](#dependencies)
3. [Data Structures](#data-structures)
4. [Class Documentation](#class-documentation)
5. [Fine-Tuning Guide](#fine-tuning-guide)
6. [Testing Recommendations](#testing-recommendations)

---

## Module Overview

The `lane_detector.py` module is responsible for detecting and tracking lane boundaries in the racing vehicle's camera feed. It performs:
- **Perspective transformation** to convert camera view to bird's-eye view
- **Image preprocessing** with adaptive contrast enhancement
- **Lane line detection** using sliding window and polynomial fitting
- **Visual odometry** for motion estimation when lanes are lost
- **Heading calculation** for vehicle orientation
- **Target position computation** for steering control

The module outputs a `LaneResult` object containing detected lanes, confidence metrics, and control guidance.

---

## Dependencies

```python
import cv2                                  # OpenCV for image processing
import numpy as np                          # NumPy for numerical operations
import math                                 # Math for trigonometric functions
from dataclasses import dataclass          # For structured result output
from perception.lane_tracker import (      # Lane tracking algorithms
    HybridLaneTracker,
    DeadReckoningNavigator
)
```

---

## Data Structures

### LaneResult (Dataclass)

Container for all lane detection outputs. Key fields:

```python
@dataclass
class LaneResult:
    warped_binary: np.ndarray          # Bird's-eye view binary image (480×640)
    lane_dbg: np.ndarray               # Debug visualization image
    sl: object                         # Left lane polynomial fit coefficients
    sr: object                         # Right lane polynomial fit coefficients
    target_x: float                    # Target X position (0-640) for steering
    lateral_error_px: float            # Deviation from center (target_x - 320)
    anchor: str                        # Detection mode identifier
    confidence: float                  # Lane detection confidence (0.0-1.0)
    lane_width_px: float               # Estimated distance between lanes (pixels)
    curvature: float                   # Road curvature at evaluation height
    heading_rad: float = 0.0           # Vehicle heading angle (radians)
    heading_conf: float = 0.0          # Confidence in heading estimate
    y_eval: float = 320.0              # Y-coordinate for lookahead evaluation
    optical_yaw_rate: float = 0.0      # Rotation rate from optical flow
    optical_vel: float = 0.0           # Forward velocity from optical flow
    signed_curvature: float = 0.0      # Signed curvature (positive=right)
    imu_yaw_rate: float = 0.0          # Rotation rate from IMU
```

---

## Class Documentation

## VisualOdometry

**Purpose**: Estimates vehicle motion (yaw rate and forward velocity) using optical flow when lane detection fails.

### Constructor: `__init__()`

```python
def __init__(self):
    # Feature detection parameters for Lucas-Kanade tracking
    self.feature_params = dict(
        maxCorners=50,          # Maximum features to track (TUNABLE)
        qualityLevel=0.3,       # Quality threshold for feature selection (0.0-1.0, TUNABLE)
        minDistance=7,          # Minimum distance between features (pixels, TUNABLE)
        blockSize=7             # Size of detection window (pixels, TUNABLE)
    )
    
    # Lucas-Kanade optical flow parameters
    self.lk_params = dict(
        winSize=(15, 15),                           # Search window size (TUNABLE)
        maxLevel=2,                                 # Pyramid levels (TUNABLE)
        criteria=(cv2.TERM_CRITERIA_EPS | 
                  cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # Convergence criteria
    )
    
    self.p0 = None                  # Previous frame features
    self.old_gray = None            # Previous frame grayscale image
```

**Fine-Tuning Parameters**:
- **maxCorners**: Increase (50→100) for more features on low-texture tracks, decrease for crowded scenes
- **qualityLevel**: Lower (0.3→0.2) to detect weaker features on worn track surfaces
- **winSize**: Increase (15,15→20,20) for slower features or low-light conditions

### Method: `update(frame_bgr, dt)`

```python
def update(self, frame_bgr, dt: float):
    # Return zero motion if dt invalid (safety check)
    if dt <= 0: return 0.0, 0.0
    
    # Convert BGR to grayscale for feature tracking
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Use only bottom 40% of image (lower perspective distortion)
    # Crops from 60% of height downward
    roi = gray[int(h * 0.6):, :]
    
    # INITIALIZATION PHASE: Find initial features
    if self.p0 is None or len(self.p0) < 10:
        # Detect features in ROI
        p0_roi = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
        
        if p0_roi is not None:
            # Adjust Y coordinates back to full image coordinates
            p0_roi[:, 0, 1] += int(h * 0.6)
            self.p0 = p0_roi
            self.old_gray = gray.copy()
        
        return 0.0, 0.0  # No motion estimate on first frame
    
    # TRACKING PHASE: Calculate optical flow between frames
    # Lucas-Kanade sparse optical flow
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        self.old_gray, gray,           # Previous and current frame
        self.p0, None,                 # Previous features and initial guess
        **self.lk_params
    )
    
    # Validity check
    if p1 is None or st is None:
        self.p0 = None
        return 0.0, 0.0
    
    # Extract successfully tracked features (status=1)
    good_new = p1[st == 1]
    good_old = self.p0[st == 1]
    
    # Default motion values
    yaw_rate = vel = 0.0
    
    # MOTION ESTIMATION (requires minimum 3+ good tracks)
    if len(good_new) > 3:
        # Calculate pixel displacements
        dx = good_new[:, 0] - good_old[:, 0]  # X displacement
        dy = good_new[:, 1] - good_old[:, 1]  # Y displacement
        
        # Yaw rate from horizontal pixel motion
        # Factor 0.015 = empirical calibration (rad/px/s)
        yaw_rate = float(-np.median(dx) * 0.015 / dt)
        
        # Forward velocity from vertical pixel motion
        # Factor 0.008 = empirical calibration (m/px/s)
        vel = float(np.median(dy) * 0.008 / dt)
    
    # Update frame for next iteration
    self.old_gray = gray.copy()
    
    # Reset features if too few remain, otherwise reshape for next iteration
    self.p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
    
    return yaw_rate, vel
```

**Key Calibration Factors** (Track Tuning):
- **0.015**: Yaw rate pixel-to-radian conversion. Increase if yaw overestimates, decrease if underestimates
- **0.008**: Velocity pixel-to-meter conversion. Increase if velocity appears too low, decrease if too high
- ROI crop `0.6`: Increase to 0.7-0.8 for elevated tracks to avoid ceiling artifacts

---

## LaneDetector

**Purpose**: Main lane detection engine combining image processing, lane tracking, and control signal generation.

### Constructor: `__init__()`

```python
def __init__(self):
    # Perspective transform source points (camera coordinates)
    # Format: [TL, TR, BL, BR] - corners of viewing area in camera image
    self.SRC_PTS = np.float32([
        [200, 260],   # Top-left: x=200 (left margin), y=260 (above hood)
        [440, 260],   # Top-right: x=440 (right margin)
        [40, 450],    # Bottom-left: x=40, y=450 (near bottom)
        [600, 450]    # Bottom-right: x=600, y=450
    ])
    
    # Perspective transform destination points (bird's-eye coordinates)
    # Maps SRC_PTS to rectangular 640×480 grid
    self.DST_PTS = np.float32([
        [150, 0],     # Top-left of output
        [490, 0],     # Top-right of output
        [150, 480],   # Bottom-left of output
        [490, 480]    # Bottom-right of output
    ])
    
    # Precomputed perspective transformation matrix
    self.M_forward = cv2.getPerspectiveTransform(self.SRC_PTS, self.DST_PTS)
    
    # Contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # clipLimit=3.0: Contrast amplification threshold (TUNABLE: 1.0-5.0)
    # tileGridSize=(8,8): Number of tiles for local histogram (TUNABLE: 4x4-16x16)
    self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    # Lane tracking module from lane_tracker.py
    self.tracker = HybridLaneTracker(img_shape=(480, 640))
    
    # Visual odometry for motion estimation
    self.vo = VisualOdometry()
    
    # Loss tracking
    self.lost_frames = 0            # Frames since last valid lane detection
    self.last_target_x = 320.0      # Fallback target (image center)
    
    # Heading smoothing (Exponential Moving Average)
    self._heading_ema = 0.0         # EMA state variable
    
    # IMU yaw tracking
    self.last_yaw = None            # Previous IMU yaw angle
```

**Calibration for Different Tracks**:

1. **SRC_PTS Adjustment** (Perspective calibration):
   - If lanes appear twisted in bird's-eye view, adjust Y coordinates
   - Example for tracks with steep angles: Change `[200, 260]` to `[200, 240]`
   
2. **CLAHE Parameters**:
   - Low-light tracks: Increase clipLimit to 4.0-5.0
   - High-contrast tracks (shadows): Decrease to 1.5-2.0
   - Tiled vs. smooth: 8×8 default, use 16×16 for uniform lighting

### Method: `process(raw_frame, dt, ...)`

**Parameter Documentation**:

```python
def process(self, 
            raw_frame,                  # BGR image from camera (any size)
            dt: float = 0.033,          # Frame time delta (seconds, ~30ms for 30fps)
            extra_offset_px=0.0,        # Additional lateral offset for steering
            nav_state="NORMAL",         # Navigation mode (NORMAL/JUNCTION_*/ROUNDABOUT)
            velocity_ms=0.0,            # Current vehicle velocity (m/s)
            last_steering=0.0,          # Previous steering angle (degrees)
            upcoming_curve: str = "STRAIGHT",  # Lane hint (STRAIGHT/LEFT/RIGHT)
            pitch_rad: float = 0.0,     # Vehicle pitch angle for dynamic correction
            current_yaw: float = 0.0    # Current vehicle heading (degrees)
) -> LaneResult:
```

**Step-by-Step Processing**:

```python
# STEP 1: Frame normalization
if raw_frame.shape[:2] != (480, 640):
    process_frame = cv2.resize(raw_frame, (640, 480))
else:
    process_frame = raw_frame
# Purpose: Ensure consistent processing dimensions

# STEP 2: Visual odometry estimation (for loss recovery)
opt_yaw_rate, opt_vel = self.vo.update(process_frame, dt)
# opt_yaw_rate: Estimated rotation from optical flow (rad/s)
# opt_vel: Estimated forward motion from optical flow (m/s)

# STEP 3: IMU yaw rate calculation
if self.last_yaw is None: self.last_yaw = current_yaw
imu_yaw_rate = (current_yaw - self.last_yaw) / max(dt, 0.001)
# Calculates rotational velocity from IMU heading angle
self.last_yaw = current_yaw

# STEP 4: Pitch-based perspective adjustment
# Compensates for vehicle pitch (climbing/descending) to prevent horizon shift
if abs(pitch_rad) > 0.001:
    # Empirical scale: 1 radian pitch ≈ 400 pixels shift
    shift_px = int(pitch_rad * 400)
    
    # Copy and adjust source points
    dyn_src = self.SRC_PTS.copy()
    dyn_src[0][1] += shift_px  # Adjust top points
    dyn_src[1][1] += shift_px
    
    # Recalculate perspective transform for this frame
    M_use = cv2.getPerspectiveTransform(dyn_src, self.DST_PTS)
else:
    M_use = self.M_forward  # Use precomputed matrix on flat terrain

# TUNING NOTE: For steep tracks, adjust factor (400 → 300-500)

# STEP 5: Perspective transformation to bird's-eye view
warped_colour = cv2.warpPerspective(process_frame, M_use, (640, 480))
# Converts camera view to top-down view for lane detection

# STEP 6: LAB color space conversion and L-channel extraction
lab = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2LAB)
L = self.clahe.apply(lab[:, :, 0])  # Apply CLAHE to L channel
# L channel is brightness-independent, better for lane detection

# STEP 7: Adaptive brightness compensation
# Problem: Different lighting conditions affect threshold
mean_l = np.mean(L)

# Dark image boost
if mean_l < 100:
    # Scale: alpha = 1.0 + adjustment up to 0.5
    # Offset: beta = brightness boost
    L = cv2.convertScaleAbs(
        L,
        alpha=1.0 + (100 - mean_l)/200,  # Gain increases as image darkens
        beta=int((100 - mean_l)*0.6)      # Brightness offset
    )
    # TUNING: For very dark tracks, use 0.8 instead of 0.6

# Bright image compensation
elif mean_l > 180:
    # Reduce contrast and darken highlights
    L = cv2.convertScaleAbs(
        L,
        alpha=1.0 - (mean_l - 180)/350,   # Reduce gain
        beta=int(-(mean_l - 180)*0.4)     # Darken
    )
    # TUNING: For very bright tracks, use 0.5 instead of 0.4

# STEP 8: Binary thresholding for lane detection
# Adaptive threshold: threshold value computed locally for each pixel
binary = cv2.adaptiveThreshold(
    L,                                   # Input L channel
    255,                                 # Maximum output value
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,      # Local mean based on Gaussian
    cv2.THRESH_BINARY_INV,               # Inverted: lanes = white (255)
    31,                                  # Neighborhood size (TUNABLE: 21-41, must be odd)
    15                                   # Constant subtracted from mean (TUNABLE: 5-25)
)
# Block size 31: Larger = more robust to local shadows, smaller = finer details
# Constant 15: Higher = easier to detect faint lanes

# TUNING FOR TRACKS:
# - Worn tracks: increase constant to 20-25, block size to 41
# - Shiny/reflective: decrease constant to 5-10, block size to 21

# STEP 9: Morphological filtering
warped_binary = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,                     # Close: dilate then erode
    cv2.getStructuringElement(           # Removes noise while preserving lanes
        cv2.MORPH_RECT, (5, 5)           # TUNABLE: 3×3-7×7
    )
)
# Purpose: Fill small holes in lanes, remove isolated noise

# STEP 10: Lane tracking update
map_hint = upcoming_curve if upcoming_curve in ("LEFT", "RIGHT") else "STRAIGHT"
sl, sr, line_dbg, mode_label = self.tracker.update(warped_binary, map_hint=map_hint)
# sl, sr: Polynomial coefficients for left/right lanes
# line_dbg: Debug visualization
# mode_label: "SLIDE" (sliding window) or "POLY" (polynomial search)

# STEP 11: Curvature calculation and dynamic lookahead
base_curv = self.tracker.get_curvature(320)  # Curvature at center

# Y-coordinate where to evaluate lane center
# Straight: y_eval = 180, Curves: y_eval increases to 300
# More lookahead in curves helps predict lane behavior
y_eval = 180.0 + min(1.0, base_curv / 0.003) * 120.0
# TUNING: Adjust 0.003 threshold (lower = sensitive to smaller curves)
# TUNING: Adjust 120.0 (lookahead depth in curves)

# STEP 12: Get target steering position
lw = self.tracker.estimated_lane_width

target_x, anchor = self.tracker.get_target_x(
    y_eval,           # Evaluation height
    lw,               # Lane width (pixels)
    extra_offset_px,  # Additional offset
    nav_state,        # Navigation context
    self.lost_frames, # Number of frames without lanes
    velocity_ms,
    last_steering,
    current_yaw,
    imu_yaw_rate=imu_yaw_rate
)

# STEP 13: Exponential Moving Average (EMA) smoothing on target
# Problem: Lane wobble at 30Hz causes steering oscillation
# Solution: Heavy smoothing (alpha=0.05) for small changes
if not hasattr(self, "_target_ema"):
    self._target_ema = target_x

delta = abs(target_x - self._target_ema)

# Adaptive smoothing: Fast for large changes, slow for micro-noise
if delta > 20.0:
    alpha = 0.7      # Large correction (curve) - respond fast
elif delta > 5.0:
    alpha = 0.3      # Moderate correction
else:
    alpha = 0.05     # Micro-noise on straights - heavy damping

self._target_ema = (1.0 - alpha) * self._target_ema + alpha * target_x
target_x = self._target_ema

# TUNING:
# - Increase alpha (0.05→0.15) if steering is too sluggish on curves
# - Decrease alpha (0.05→0.02) if steering oscillates

# STEP 14: Loss tracking and dead reckoning fallback
if target_x is None:
    self.lost_frames += 1
    # Use IMU to estimate position when lines are lost
    self.tracker.dead_reckoner.accumulate(dt, current_yaw)
    target_x = self.last_target_x  # Use previous target
else:
    self.lost_frames = 0
    self.last_target_x = target_x

# STEP 15: Calculate curvature metrics
curv = self.tracker.get_curvature(y_eval)
signed_curv = self.tracker.get_signed_curvature(y_eval)

# Confidence estimation
conf = 1.0 if (sl is not None and sr is not None) else \
       0.5 if (sl is not None or sr is not None) else 0.0

# STEP 16: Heading calculation (vehicle orientation relative to lane)
heading_rad = 0.0

def _lane_heading(fit, y):
    """Calculate heading angle from polynomial fit"""
    # Tangent angle = atan2(Δx / Δy)
    return math.atan2(
        np.polyval(fit, y - 50) - np.polyval(fit, y),  # X change
        50  # Y change (50 pixels)
    )

# Average heading from both lanes if available
if sl is not None and sr is not None:
    heading_rad = (_lane_heading(sl, y_eval) + _lane_heading(sr, y_eval)) / 2.0
elif sl is not None:
    heading_rad = _lane_heading(sl, y_eval)
elif sr is not None:
    heading_rad = _lane_heading(sr, y_eval)

# Heavy EMA smoothing on heading (α=0.85) to prevent reactive wobble
self._heading_ema = 0.85 * self._heading_ema + 0.15 * heading_rad
heading_rad = self._heading_ema

# STEP 17: Return complete result
return LaneResult(
    warped_binary=warped_binary,
    lane_dbg=line_dbg,
    sl=sl, sr=sr,
    target_x=target_x,
    lateral_error_px=target_x - 320.0,  # Deviation from center
    anchor=anchor,                        # Detection source label
    confidence=conf,
    lane_width_px=lw,
    curvature=curv,
    heading_rad=heading_rad,
    heading_conf=conf,
    y_eval=y_eval,
    optical_yaw_rate=opt_yaw_rate,
    optical_vel=opt_vel,
    signed_curvature=signed_curv,
    imu_yaw_rate=imu_yaw_rate,
)
```

---

## Fine-Tuning Guide

### Track Testing Checklist

| Issue | Symptom | Adjustment | Impact |
|-------|---------|-----------|--------|
| **Lanes lost in curves** | `confidence=0.0` frequently | Increase `base_curv` threshold (0.003→0.002) | More aggressive lookahead |
| **Steering oscillates** | Wobble on straights | Decrease EMA alpha (0.05→0.02) | Smoother output |
| **Late response to curves** | Car doesn't turn until late | Increase alpha (0.05→0.15) for large deltas | Faster steering |
| **Dark track issues** | Binary image all black | Increase CLAHE clipLimit (3.0→4.5) | Better contrast in shadows |
| **Bright track washed out** | Binary image all white | Decrease bright compensation (0.4→0.3) | Better distinction |
| **Perspective distortion** | Lanes curved unnaturally | Adjust `SRC_PTS` Y values (±20px) | Better registration |
| **Lane width varies wildly** | Unreliable `lane_width_px` | Increase EMA_ALPHA in tracker (0.85→0.95) | Smoother width estimate |

### Parameter Sensitivity Analysis

**High Priority (Most Impact)**:
1. `adaptive threshold block size` (31) - ±10 affects detection
2. `pitch compensation factor` (400) - ±100 on elevated tracks
3. `CLAHE clipLimit` (3.0) - ±2.0 on different lighting
4. `EMA alpha` (0.05) - ±0.05 on oscillation control

**Medium Priority**:
5. `y_eval lookahead` (180-300) - Affects curve response
6. `brightness compensation factors` (0.6, 0.4) - ±0.2
7. `morphology kernel` (5×5) - ±2 for noise filtering

**Low Priority (Marginal)**:
8. Visual odometry factors (0.015, 0.008) - ±0.003
9. Heading EMA (0.85) - ±0.1
10. Lane width thresholds - Rarely need adjustment

---

## Testing Recommendations

### Phase 1: Baseline Testing
```
1. Run on reference track with good lighting
2. Record: confidence, target_x stability, heading_rad smoothness
3. Establish baseline metrics (e.g., target_x std dev < 10px)
```

### Phase 2: Adverse Conditions
```
Test scenarios:
- Early morning (low light, long shadows)
- Midday (high contrast, reflections)
- Late afternoon (low angle sun)
- Overcast (uniform gray lighting)
- After rain (wet reflections)
```

### Phase 3: Track-Specific Tuning
```
For each track:
1. Capture 30-second baseline video
2. Analyze histogram distribution in L channel
3. Adjust CLAHE parameters if mean_L consistently outside [80-150]
4. Run same video segment 5 times, measure target_x variance
5. Adjust EMA alphas if variance > threshold
6. Document final parameters for replay
```

### Performance Metrics
```python
# Calculate stability (lower is better)
target_x_values = [...]  # From 100 frames
stability = np.std(target_x_values)  # Should be < 5px on straights

# Confidence tracking
confidence_history = [...]
loss_rate = sum(1 for c in confidence_history if c < 0.3) / len(confidence_history)
# Should be < 5% on well-marked tracks

# Response time to curves
# Measure delay between curve_hint and y_eval increase
# Target: < 500ms lag
```

---

## Summary: Key Parameters for Track Tuning

```python
# In LaneDetector.__init__():
SRC_PTS adjustment        # Perspective calibration (primary)
CLAHE clipLimit           # Low-light vs. bright balance
clahe tileGridSize        # Contrast distribution

# In LaneDetector.process():
adaptive threshold block   # Detection sensitivity (31)
brightness compensation   # Exposure compensation (0.6, 0.4)
morphology kernel         # Noise filtering (5×5)
y_eval lookahead formula  # Curve response (120.0 depth)
EMA alpha values          # Steering smoothness (0.05, 0.7, 0.3)

# In VisualOdometry.__init__():
feature optical flow params # Fallback quality when lanes lost
calibration factors       # Yaw (0.015) and velocity (0.008)
```

