# Lane Tracker Module - Comprehensive Technical Documentation

## Table of Contents
1. [Module Overview](#module-overview)
2. [Class Documentation](#class-documentation)
3. [Algorithm Details](#algorithm-details)
4. [Fine-Tuning Parameters](#fine-tuning-parameters)
5. [Track Testing Guide](#track-testing-guide)
6. [Troubleshooting](#troubleshooting)

---

## Module Overview

The `lane_tracker.py` module implements advanced lane tracking and dead reckoning algorithms:

- **HybridLaneTracker**: Dual-mode lane detection (sliding window for initial search, polynomial band for tracking)
- **DeadReckoningNavigator**: IMU-based fallback navigation when lanes are lost

The module handles:
- Lane polynomial fitting with exponential smoothing
- Width validation and sanity checks
- Multiple navigation states (NORMAL, JUNCTION, ROUNDABOUT)
- IMU-based dead reckoning for lane loss recovery
- Curvature calculation and signed curvature

---

## Class Documentation

## Class: DeadReckoningNavigator

**Purpose**: Provides fallback navigation using IMU heading when lane detection fails.

### Constructor: `__init__()`

```python
def __init__(self):
    self.last_valid_target = 320.0      # Last known good steering target (image center X)
    self.last_valid_curvature = 0.0     # Last known road curvature (for curved roads)
    self._lost_time_s = 0.0             # Time elapsed since lanes were lost (seconds)
    self.yaw_at_loss = 0.0              # Vehicle heading when lanes were lost (degrees)
    self.is_lost = False                # Flag indicating lane loss state
```

**State Variables Explained**:
- `last_valid_target`: Stores the last target X position before loss. If next frame finds lanes again, this prevents sudden jumps.
- `_lost_time_s`: Accumulates frame-to-frame (tracks how long has been blind)
- `yaw_at_loss`: Records the exact heading when loss occurred (reference point for corrections)

### Method: `reset_lost_timer(current_yaw)`

```python
def reset_lost_timer(self, current_yaw: float):
    """Called when lanes are detected again to reset loss tracking"""
    
    self._lost_time_s = 0.0         # Reset loss duration counter
    self.yaw_at_loss = current_yaw  # Store baseline heading for next loss
    self.is_lost = False            # Clear loss flag
```

**When Called**: After `predict_target()` returns valid lanes
**Effect**: Resets all loss timers for next detection cycle

### Method: `accumulate(dt, current_yaw)`

```python
def accumulate(self, dt: float, current_yaw: float):
    """Called each frame while lanes are lost; accumulates time"""
    
    # On first loss frame, capture the reference heading
    if not self.is_lost:
        self.yaw_at_loss = current_yaw  # Store baseline heading
        self.is_lost = True             # Set lost flag
    
    # Accumulate elapsed time
    self._lost_time_s += dt
```

**Called Per Frame**: Yes, when `target_x is None` in lane_detector.py
**Accumulation**: Each 30ms frame adds 0.03s to counter

### Method: `predict_target(last_speed, last_steering, current_yaw)`

```python
def predict_target(self, last_speed, last_steering, current_yaw):
    """
    Predicts steering target based on:
    - Last valid curvature (is road curved or straight?)
    - IMU yaw drift (is car drifting left or right?)
    
    Returns: (predicted_target_x, confidence)
    """
    
    # Time since loss began
    t = max(0.0, self._lost_time_s)
    
    # Calculate how much the car has rotated since loss
    # positive = rotating right, negative = rotating left
    delta_yaw_deg = current_yaw - self.yaw_at_loss
    
    # DECISION: Curved road vs. Straight road behavior
    if abs(self.last_valid_curvature) > 0.0015 or abs(last_steering) > 5.0:
        # CURVED ROAD MODE
        # Problem: If lanes drop mid-curve, we should maintain the curve
        # Solution: Keep using the last target position (which was optimal for the curve)
        
        predicted_target = self.last_valid_target
        
        # Confidence decays over time (3 second timeout for curves)
        # 100% → 50% at 1.5s → 0% at 3s
        confidence = max(0.0, 1.0 - t / 3.0)
        
        # TUNING NOTE:
        # - Increase 3.0 → 4.0 for longer curves (gentle bends)
        # - Decrease 3.0 → 2.0 for tight curves (chicanes)
        
    else:
        # STRAIGHT ROAD MODE
        # Problem: Car drifts left/right due to wind, uneven surface, steering hysteresis
        # Solution: Use IMU to detect drift and counteract it
        
        # CRITICAL LOGIC (explained below):
        # If car rotates right (+5 deg), it's drifting RIGHT
        # To correct, we steer LEFT by moving target LEFT (< 320)
        # Formula: predicted_target = 320 - (delta_yaw * 20)
        # Example: delta_yaw = +5 deg → target = 320 - 100 = 220 (steers LEFT)
        
        predicted_target = 320.0 - (delta_yaw_deg * 20.0)
        
        # Confidence decays more slowly on straights (5 second timeout)
        # 100% → 50% at 2.5s → 0% at 5s
        confidence = max(0.0, 1.0 - t / 5.0)
        
        # TUNING NOTE:
        # Factor 20.0: Gain applied to yaw error (pixels per degree)
        # - Increase 20.0 → 25.0 if car drifts despite correction
        # - Decrease 20.0 → 15.0 if correction overshoots
        
        # TUNING NOTE:
        # Timeout 5.0: Maximum time to navigate blind on straight
        # - Increase to 7.0 for long straightaways
        # - Decrease to 3.0 if multiple tight curves expected
    
    # Clamp target to valid image bounds [150, 490]
    predicted_target = float(np.clip(predicted_target, 150, 490))
    
    return predicted_target, confidence
```

**Dead Reckoning Logic Explained**:

The yaw-based correction works because:
1. **If delta_yaw is +5° (right)**: Car has rotated right → drifting right → steer left
2. **Pixel domain**: Steering left = target moves left (< 320)
3. **Formula**: `320 - (delta_yaw * gain)` creates inverse feedback

**Tuning for Tracks**:
- **High-speed straights**: Increase timeout (5.0→7.0), decrease gain (20→15)
- **Low-friction surfaces**: Increase gain (20→25) for aggressive correction
- **Wind-prone tracks**: Increase timeout (5.0→6.0) to maintain heading longer

---

## Class: HybridLaneTracker

**Purpose**: Main lane detection engine using two complementary modes:
1. **Sliding Window Mode**: Initial search when lanes are lost
2. **Polynomial Search Mode**: Fast tracking around previous polynomials

### Class Variables (Constants)

```python
# Sliding window search parameters
NWINDOWS = 9                # Number of horizontal bands to search (TUNABLE: 5-12)
SW_MARGIN = 60              # Search width around sliding window center (TUNABLE: 40-100)
MINPIX = 50                 # Minimum pixels to recenter window (TUNABLE: 20-100)

# Polynomial search band margin (varies with curvature)
POLY_MARGIN_BASE = 60       # Margin on straight roads (pixels, TUNABLE: 40-80)
POLY_MARGIN_CURV = 120      # Margin on curves (pixels, TUNABLE: 80-160)
MIN_PIX_OK = 200            # Minimum pixels for valid lane (TUNABLE: 100-300)

# Exponential Moving Average smoothing
EMA_ALPHA = 0.85            # Smoothing on straights (TUNABLE: 0.7-0.95)
EMA_ALPHA_TURN = 1.0        # No smoothing during sharp turns (TUNABLE: 0.8-1.0)
# Note: EMA_ALPHA_TURN=1.0 means α_new only, no smoothing

# Stale frame tracking
STALE_FIT_FRAMES = 12       # Frames before forgetting lane (TUNABLE: 8-20)
# At 30fps ≈ 400ms timeout for lane persistence

# Lane geometry constraints
WIDE_ROAD_PX = 420          # Maximum lane width (TUNABLE: 350-500)
SINGLE_LANE_PX = 200        # Minimum lane width (TUNABLE: 150-250)
RIGHT_LANE_BIAS_PX = 0      # Offset for right lane preference (TUNABLE: -40 to +40)
DIVIDER_FOLLOW_OFFSET_PX = 145  # Offset when following divider line (TUNABLE: 100-200)
```

**Key Tuning Insights**:
- Increase NWINDOWS (9→12) for roads with many lane markings
- Increase MIN_PIX_OK (200→300) for worn tracks with faint markings
- Decrease STALE_FIT_FRAMES (12→8) for tracks with frequent lane changes
- Adjust WIDE_ROAD_PX based on actual track width in pixels

### Constructor: `__init__(img_shape=(480, 640))`

```python
def __init__(self, img_shape=(480, 640)):
    # Image dimensions (from perspective-transformed view)
    self.h, self.w = img_shape  # h=480 (height), w=640 (width)
    
    # Operating mode indicator
    self.mode = "SEARCH"  # "SEARCH" = sliding window, "TRACKING" = polynomial band
    
    # Polynomial coefficients for left and right lanes
    # fit: [a, b, c] representing f(y) = ay² + by + c (x-coordinate as function of y)
    self.left_fit = None    # Raw polynomial from current frame
    self.right_fit = None
    
    # Exponential Moving Average smoothed polynomials (main output)
    self.sl = None          # Smoothed left polynomial
    self.sr = None          # Smoothed right polynomial
    
    # Pixel count confidence metrics
    self.left_conf = 0      # Number of pixels assigned to left lane
    self.right_conf = 0
    
    # Stale frame counters (tracks how long since last valid detection)
    self.left_stale = 0     # Incremented each frame without valid left lane
    self.right_stale = 0
    
    # Estimated distance between lane markings (pixels)
    self.estimated_lane_width = 280.0  # (TUNABLE: 200-350 depending on track)
    
    # Loss tracking for right lane (for dead reckoning)
    self.right_lost_frames = 0
    
    # IMU-based fallback navigation when both lanes lost
    self.dead_reckoner = DeadReckoningNavigator()
```

**State Machine**:
- Mode changes: `SEARCH` ↔ `TRACKING`
- Transition to `TRACKING` when: `has_l or has_r or sl is not None or sr is not None`
- Transition to `SEARCH` when: All conditions fail

### Method: `update(warped_binary, map_hint="STRAIGHT")`

```python
def update(self, warped_binary, map_hint: str = "STRAIGHT"):
    """
    Main update function called each frame with binary lane image
    
    Args:
        warped_binary: Bird's-eye view binary image (0/255)
        map_hint: "STRAIGHT", "LEFT", or "RIGHT" for expected curve direction
    
    Returns:
        sl, sr: Smoothed polynomial coefficients (or None if no lanes)
        dbg: Debug visualization image
        mode_label: "SLIDE" (sliding window) or "POLY" (polynomial search)
    """
    
    # Extract non-zero pixel coordinates
    nz = warped_binary.nonzero()        # Get all white pixels (255)
    nzy = np.array(nz[0])               # Y coordinates
    nzx = np.array(nz[1])               # X coordinates
    
    # MODE SELECTION: Sliding window vs. polynomial band search
    if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
        # TRACKING MODE: We have previous polynomials
        # Use fast polynomial band search
        curv = self.get_curvature(self.h // 2)
        li, ri, dbg = self._poly_search(warped_binary, nzx, nzy, curvature=curv, map_hint=map_hint)
        mode_label = "POLY"
    else:
        # SEARCH MODE: Lost lanes or first frame
        # Use sliding window (slower but more robust)
        li, ri, dbg = self._sliding_window(warped_binary, nzx, nzy, map_hint=map_hint)
        mode_label = "SLIDE"
    
    # LANE VALIDATION: Check if detected pixels meet threshold
    self.left_conf = len(li)   # Confidence = number of pixels
    self.right_conf = len(ri)
    has_l = self.left_conf >= self.MIN_PIX_OK    # At least 200 pixels
    has_r = self.right_conf >= self.MIN_PIX_OK
    
    # LEFT LANE PROCESSING
    if has_l:
        # Fit polynomial: x = ay² + by + c
        fl = np.polyfit(nzy[li], nzx[li], 2)  # Quadratic fit
        self.left_fit = fl  # Raw fit from this frame
        
        # Check if we're in a sharp turn (curvature > 0.002)
        curv_now = self.get_curvature(self.h // 2)
        alpha = self.EMA_ALPHA_TURN if curv_now > 0.002 else self.EMA_ALPHA
        # Sharp turn: α=1.0 (respond instantly)
        # Straight: α=0.85 (smooth out noise)
        
        # Apply exponential moving average
        # sl_new = α × fl_new + (1-α) × fl_old
        self.sl = self._ema(self.sl, fl, alpha)
        
        # Reset stale counter (we just found the lane)
        self.left_stale = 0
    else:
        # No valid left lane this frame
        self.left_stale += 1  # Increment stale counter
        
        # If lane has been missing for too long, forget about it
        if self.left_stale > self.STALE_FIT_FRAMES:  # Default 12 frames
            self.left_fit, self.sl = None, None
    
    # RIGHT LANE PROCESSING (same logic as left)
    if has_r:
        fr = np.polyfit(nzy[ri], nzx[ri], 2)
        self.right_fit = fr
        curv_now = self.get_curvature(self.h // 2)
        alpha = self.EMA_ALPHA_TURN if curv_now > 0.002 else self.EMA_ALPHA
        self.sr = self._ema(self.sr, fr, alpha)
        self.right_stale = 0
    else:
        self.right_stale += 1
        if self.right_stale > self.STALE_FIT_FRAMES:
            self.right_fit, self.sr = None, None
    
    # DUAL-LANE SANITY CHECK
    # Problem: Sometimes noise creates "phantom lanes" with unrealistic width
    # Solution: If both lanes found but width is invalid, discard the weaker one
    if has_l and has_r:
        if not self._width_sane(self.left_fit, self.right_fit):
            # Lane width is outside [180, 420] pixels
            # Discard the lane with fewer pixels (less confident)
            if self.left_conf < self.right_conf:
                self.left_fit, self.sl, self.left_stale, has_l = None, None, self.STALE_FIT_FRAMES, False
            else:
                self.right_fit, self.sr, self.right_stale, has_r = None, None, self.STALE_FIT_FRAMES, False
        else:
            # Both lanes valid and reasonable width
            # Update estimated lane width using weighted average at different heights
            y_positions = [100, 200, 300, 400]  # Four sample heights
            widths = [np.polyval(self.sr, y) - np.polyval(self.sl, y) for y in y_positions]
            
            # Weight higher positions (closer to camera) more heavily
            weighted_avg_width = np.average(widths, weights=[4, 3, 2, 1])
            
            # EMA update of width estimate
            # 80% previous + 20% new (heavily smoothed)
            self.estimated_lane_width = 0.8 * self.estimated_lane_width + 0.2 * weighted_avg_width
            
            # TUNING: Change 0.8/0.2 ratio:
            # - 0.9/0.1 for more stable width (slower adaptation)
            # - 0.7/0.3 for faster width tracking
    
    # MODE TRANSITION LOGIC
    # Transition to TRACKING if we have at least one good lane
    self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"
    
    return self.sl, self.sr, dbg, mode_label
```

**Key Processing Steps**:
1. Extract white pixels (lanes)
2. Choose search algorithm based on mode
3. Validate pixel counts
4. Fit polynomials to each lane
5. Apply EMA smoothing with curvature-sensitive alpha
6. Update stale counters
7. Validate lane width sanity
8. Update mode for next frame

### Method: `get_target_x(y_eval, lane_width_px, ...)`

This is the **critical method** for steering control. It handles different navigation modes:

```python
def get_target_x(self, y_eval, lane_width_px, extra_offset_px=0,
                 nav_state="NORMAL", frames_lost=0,
                 last_speed=0.0, last_steering=0.0, current_yaw=0.0,
                 imu_yaw_rate=0.0):
    """
    Computes the target X-position for steering control.
    Handles multiple navigation contexts with fallback strategies.
    
    Returns: (target_x, anchor_description)
    """
    
    # Shorthand for polynomials
    sl, sr = self.sl, self.sr
    hw = lane_width_px / 2.0  # Half-width for offset calculations
    
    # Helper function to evaluate polynomial at height y
    def ev(fit): 
        return float(np.polyval(fit, y_eval))
    
    # ============================================================
    # CASE 1: ROUNDABOUT MODE
    # ============================================================
    if nav_state == "ROUNDABOUT":
        # In roundabout: hug inner lane to minimize turning radius
        
        if sl is not None:
            # Inner lane available (left lane is inner)
            return ev(sl) + hw + extra_offset_px, "RBT_INNER"
        if sr is not None:
            # Outer lane available (right lane is inner)
            return ev(sr) - hw + extra_offset_px, "RBT_OUTER"
        
        # Both lanes lost in roundabout (dangerous)
        return None, "RBT_LOST"
    
    # ============================================================
    # CASE 2: JUNCTION MODE (3 sub-cases)
    # ============================================================
    if nav_state.startswith("JUNCTION"):
        if nav_state == "JUNCTION_RIGHT":
            # Turning right at junction
            if sr is not None:
                # Right lane available: use right edge for sharp turn
                return ev(sr) - (lane_width_px * 0.40) + extra_offset_px, "JCT_RIGHT_EDGE"
            elif sl is not None:
                # Only left lane available: project across lane
                return ev(sl) + (lane_width_px * 1.5) + extra_offset_px, "JCT_RIGHT_GHOST"
            else:
                # No lanes visible: use blind estimate
                return 320.0 + (lane_width_px * 0.8) + extra_offset_px, "JCT_RIGHT_BLIND"
        
        elif nav_state == "JUNCTION_LEFT":
            # Turning left at junction (mirror of right logic)
            if sl is not None:
                return ev(sl) + (lane_width_px * 0.40) + extra_offset_px, "JCT_LEFT_EDGE"
            elif sr is not None:
                return ev(sr) - (lane_width_px * 1.5) + extra_offset_px, "JCT_LEFT_GHOST"
            else:
                return 320.0 - (lane_width_px * 0.8) + extra_offset_px, "JCT_LEFT_BLIND"
        
        # Still waiting for user input (parking, waiting)
        return 320.0 + extra_offset_px, "JCT_WAITING_CHOICE"
    
    # ============================================================
    # CASE 3: NORMAL MODE (main driving)
    # ============================================================
    
    # Check lane availability
    has_right = (sr is not None)
    has_left = (sl is not None)
    
    # CASE 3.1: BOTH LANES LOST → Dead reckoning
    if not has_right and not has_left:
        # Use IMU heading to predict target (see DeadReckoningNavigator)
        predicted_x, conf = self.dead_reckoner.predict_target(last_speed, last_steering, current_yaw)
        return predicted_x + extra_offset_px, f"DEAD_RECKONING_{conf:.2f}"
    
    # CASE 3.2: RIGHT LANE AVAILABLE
    if has_right:
        # Reset right-lane loss tracking
        self.right_lost_frames = 0
        self.right_yaw_at_loss = current_yaw
        
        if has_left:
            # DUAL LANE: Target center between lanes
            base_x = (ev(sl) + ev(sr)) / 2.0
            anchor = "CENTER_DUAL"
        else:
            # RIGHT LANE ONLY: Target inner edge of right lane
            base_x = ev(sr) - hw
            anchor = "RIGHT_LANE_ONLY"
    
    # CASE 3.3: ONLY LEFT LANE (right lost)
    else:
        # Increment loss counter for right lane
        self.right_lost_frames += 1
        
        if has_left:
            # Can still see left lane while right is lost
            # Use IMU to estimate position of missing right lane
            
            # Predict where right lane should be based on yaw
            # When right lane disappeared, save yaw and add 15° offset
            target_yaw = getattr(self, 'right_yaw_at_loss', current_yaw) + 15.0
            delta_yaw_deg = current_yaw - target_yaw
            
            # Use IMU drift correction to find right lane position
            base_x = 320.0 - (delta_yaw_deg * 20.0)
            anchor = "IMU_15_DEG_DEAD_RECKONING"
        
        elif self.right_lost_frames < 80:  # ~4 seconds at 20 Hz
            # BOTH LANES LOST RECENTLY: Use IMU heading only
            # Steer 5 degrees left to recover lanes
            target_yaw = getattr(self, 'right_yaw_at_loss', current_yaw) - 5.0
            delta_yaw_deg = current_yaw - target_yaw
            
            # Map yaw deviation to pixel target
            # +5° yaw (drifted right) → target = 320 - 100 = 220 (steer left)
            base_x = 320.0 - (delta_yaw_deg * 20.0)
            anchor = "IMU_5_DEG_LEFT_FALLBACK"
            
            # TUNING for track:
            # Change -5.0 to different values:
            # -3.0: Gentle sweeping search
            # -7.0: Aggressive search (risky if sharp corner approaching)
        
        else:
            # BOTH LANES LOST FOR 4+ SECONDS: Give up and use dead reckoning
            predicted_x, conf = self.dead_reckoner.predict_target(last_speed, last_steering, current_yaw)
            return predicted_x + extra_offset_px, f"DEAD_RECKONING_{conf:.2f}"
    
    # ============================================================
    # FINAL ADJUSTMENTS (applied to all successful cases)
    # ============================================================
    
    # IMU yaw rate damping: Apply lateral correction based on rotation rate
    # If car is rotating right (yaw_rate > 0), move target left
    # Factor 5.0: pixels per rad/s (TUNABLE: 3.0-8.0)
    base_x -= (imu_yaw_rate * 5.0)
    
    # Save state for dead reckoning (in case next frame loses lanes)
    self.dead_reckoner.last_valid_target = base_x
    self.dead_reckoner.last_valid_curvature = self.get_curvature(y_eval)
    self.dead_reckoner.reset_lost_timer(current_yaw)
    
    return base_x + extra_offset_px, anchor
```

**Navigation Mode Behavior Summary**:

| Mode | Behavior | Fallback |
|------|----------|----------|
| NORMAL (dual lanes) | Center between lanes | Use one lane |
| NORMAL (single lane) | Target inner edge | IMU prediction |
| NORMAL (both lost <4s) | 5° left steering sweep | Dead reckon |
| NORMAL (both lost >4s) | Full IMU dead reckoning | Straight (320.0) |
| ROUNDABOUT | Hug inner lane | None (error) |
| JUNCTION_LEFT/RIGHT | Track corresponding edge | Blind estimate |

### Method: `get_curvature(y_eval)`

```python
def get_curvature(self, y_eval):
    """
    Calculates unsigned (absolute) curvature of road at height y_eval.
    
    Curvature formula: κ = |d²x/dy²| / (1 + (dx/dy)²)^(3/2)
    For quadratic fit f(y) = ay² + by + c:
        κ = |2a| / (1 + (2ay + b)²)^(3/2)
    
    Returns: Positive scalar (0 = straight, higher = more curved)
    """
    
    # Use right lane if available (more stable), otherwise left
    fit = self.sr if self.sr is not None else self.sl
    
    if fit is None: 
        return 0.0  # No lane detected
    
    # Extract quadratic coefficient and linear coefficient
    a, b = fit[0], fit[1]
    
    # Calculate denominator: (1 + (dy/dy)²)^(3/2)
    # dy/dy = 2ay + b (derivative of quadratic)
    denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
    
    # Calculate curvature (absolute value)
    return abs(2.0 * a) / max(denom, 1e-6)
```

**Interpretation**:
- Curvature = 0.0: Straight road
- Curvature = 0.001-0.002: Gentle curve
- Curvature = 0.002-0.005: Moderate curve (typical cornering)
- Curvature > 0.005: Sharp curve (hairpin)

**Used For**:
1. Lookahead distance adjustment (higher curve = look farther)
2. EMA smoothing factor selection (sharp turn = no smoothing)
3. Polynomial search margin (wider margin for curves)

### Method: `get_signed_curvature(y_eval)`

```python
def get_signed_curvature(self, y_eval):
    """
    Calculates SIGNED curvature (includes direction).
    
    Positive = right-turning curve
    Negative = left-turning curve
    """
    
    fit = self.sr if self.sr is not None else self.sl
    if fit is None: 
        return 0.0
    
    a, b = fit[0], fit[1]
    denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
    
    # Remove absolute value to preserve sign
    return (2.0 * a) / max(denom, 1e-6)
```

**Sign Convention**:
- Positive a coefficient → road curves right
- Negative a coefficient → road curves left

### Method: `_sliding_window(warped, nzx, nzy, map_hint="STRAIGHT")`

```python
def _sliding_window(self, warped, nzx, nzy, map_hint: str = "STRAIGHT"):
    """
    Sliding window search: robust initial lane detection.
    
    Algorithm:
    1. Divide image into NWINDOWS horizontal bands
    2. Find peaks in each band's histogram
    3. Track laterally as we move up the image
    4. Return indices of pixels belonging to each lane
    """
    
    # Debug visualization
    dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    
    # Histogram of bottom half (closest to camera, highest confidence)
    hist = np.sum(warped[self.h // 2:, :], axis=0)  # Sum over rows
    
    # Expected split point (40% = left margin, 60% = right starts)
    mid = int(self.w * 0.40)  # Around x=256 for 640px width
    margin = self.SW_MARGIN   # ±60 pixels search range
    
    # CURVE ADAPTATION: Adjust initial positions based on map hint
    shift = 0
    if map_hint == "LEFT":
        shift = -80  # Expect left lane further left
    elif map_hint == "RIGHT":
        shift = 80   # Expect right lane further right
    
    # Initial search ranges
    l_lo = max(margin, margin + shift)                    # Left search lower bound
    l_hi = max(l_lo + 1, mid - margin + shift)            # Left search upper bound
    r_lo = max(margin, mid + margin + shift)              # Right search lower bound
    r_hi = min(self.w - margin, self.w - margin)          # Right search upper bound
    
    # Find histogram peaks in each range
    lb = int(np.argmax(hist[l_lo:l_hi])) + l_lo if l_hi > l_lo else margin
    rb = int(np.argmax(hist[r_lo:r_hi])) + r_lo if r_hi > r_lo else mid + margin
    
    # SANITY CHECK: If peaks too close, use smoothing to find better split
    if abs(rb - lb) < 100:  # Less than 100px apart (unrealistic)
        # Apply moving average to smooth histogram
        smoothed = np.convolve(hist.astype(float), np.ones(20) / 20, mode='same')
        
        # Find first and second peaks
        p1 = int(np.argmax(smoothed))
        tmp = smoothed.copy()
        tmp[max(0, p1-40):min(self.w, p1+40)] = 0  # Suppress first peak
        p2 = int(np.argmax(tmp))
        
        # Assign peaks to left/right
        lb, rb = (min(p1, p2), max(p1, p2))
    
    # Initialize pixel lists for each lane
    wh = self.h // self.NWINDOWS  # Height of each window
    lx, rx = lb, rb              # Current X position (center)
    li, ri = [], []              # Pixel indices for left/right
    
    # SLIDING WINDOW LOOP: Scan from bottom to top
    for win in range(self.NWINDOWS):
        # Calculate window bounds (bottom-to-top scan)
        y_lo = self.h - (win + 1) * wh
        y_hi = self.h - win * wh
        
        # X bounds around current center
        xl0, xl1 = max(0, lx - self.SW_MARGIN), min(self.w, lx + self.SW_MARGIN)
        xr0, xr1 = max(0, rx - self.SW_MARGIN), min(self.w, rx + self.SW_MARGIN)
        
        # Draw debug rectangles (visualization)
        cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 255, 0), 2)
        cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 255, 0), 2)
        
        # Extract pixels in this window for each lane
        gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0) & (nzx < xl1)).nonzero()[0]
        gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0) & (nzx < xr1)).nonzero()[0]
        
        li.append(gl)
        ri.append(gr)
        
        # Recenter window if enough pixels found
        if len(gl) > self.MINPIX:
            lx = int(np.mean(nzx[gl]))
        if len(gr) > self.MINPIX:
            rx = int(np.mean(nzx[gr]))
    
    # Concatenate all window pixels
    li = np.concatenate(li) if len(li) else np.array([])
    ri = np.concatenate(ri) if len(ri) else np.array([])
    
    # Color code pixels in debug image
    if len(li):
        dbg[nzy[li], nzx[li]] = [255, 80, 80]  # Red = left lane
    if len(ri):
        dbg[nzy[ri], nzx[ri]] = [80, 80, 255]  # Blue = right lane
    
    return li, ri, dbg
```

**Performance Tuning**:
- **NWINDOWS**: More windows = finer tracking but slower
- **SW_MARGIN**: Larger = more robust to sharp curves but slower
- **map_hint adjustment**: Critical for turns (LEFT/RIGHT expected)

### Method: `_poly_search(warped, nzx, nzy, curvature=0.0, map_hint="STRAIGHT")`

```python
def _poly_search(self, warped, nzx, nzy, curvature=0.0, map_hint: str = "STRAIGHT"):
    """
    Fast polynomial band search around previously detected lanes.
    
    Algorithm:
    1. Create band of margin around sl and sr polynomials
    2. Find all pixels within band
    3. If pixels insufficient, fallback to sliding window
    """
    
    dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    
    # Margin depends on curvature (wider margin for curved roads)
    m = (self.POLY_MARGIN_CURV if curvature > 0.0015 else self.POLY_MARGIN_BASE)
    # Straight: 60px margin, Curves: 120px margin
    
    # Band search function: find pixels within ±m of polynomial
    def band(fit):
        return ((nzx > np.polyval(fit, nzy) - m) & 
                (nzx < np.polyval(fit, nzy) + m)).nonzero()[0]
    
    # Search around polynomials
    li = band(self.sl) if self.sl is not None else np.array([], dtype=int)
    ri = band(self.sr) if self.sr is not None else np.array([], dtype=int)
    
    # Fallback: if not enough pixels, return to sliding window
    if len(li) < self.MIN_PIX_OK or len(ri) < self.MIN_PIX_OK:
        self.mode = "SEARCH"
        return self._sliding_window(warped, nzx, nzy, map_hint=map_hint)
    
    # Color code debug image
    if len(li):
        dbg[nzy[li], nzx[li]] = [255, 80, 80]
    if len(ri):
        dbg[nzy[ri], nzx[ri]] = [80, 80, 255]
    
    return li, ri, dbg
```

**Performance**: ~5× faster than sliding window when lanes stable.

### Method: `_width_sane(lf, rf, y=400)`

```python
def _width_sane(self, lf, rf, y=400):
    """
    Validates lane width at height y.
    Rejects phantom lanes from noise.
    
    Valid range: 180-420 pixels (tunable)
    """
    if rf is None or lf is None:
        return False
    
    w = np.polyval(rf, y) - np.polyval(lf, y)  # Width = right - left
    
    # Check against bounds
    return self.SINGLE_LANE_PX < w < self.WIDE_ROAD_PX
    # Default: 200 < w < 420
```

**Tuning**: Adjust bounds if track has unusual geometry.

### Method: `_ema(prev, new, alpha=None)`

```python
def _ema(self, prev, new, alpha=None):
    """
    Exponential Moving Average: α×new + (1-α)×prev
    
    High α (0.9): Respond fast, more noise
    Low α (0.1): Smooth, lag on changes
    """
    
    if alpha is None:
        alpha = self.EMA_ALPHA  # Default 0.85
    
    if prev is None:
        return new.copy()  # First frame
    
    # Blend new and previous
    return alpha * new + (1.0 - alpha) * prev
```

---

## Fine-Tuning Parameters

### Priority 1 (Highest Impact)

| Parameter | Default | Range | When to Tune | Effect |
|-----------|---------|-------|--------------|--------|
| MIN_PIX_OK | 200 | 100-300 | Worn/faint lanes | Detection sensitivity |
| NWINDOWS | 9 | 5-12 | Many markings | Spatial resolution |
| SW_MARGIN | 60 | 40-100 | Tight curves | Search robustness |
| EMA_ALPHA | 0.85 | 0.7-0.95 | Oscillation | Output smoothness |
| POLY_MARGIN_BASE | 60 | 40-80 | Track geometry | Tracking bandwidth |

### Priority 2 (Medium Impact)

| Parameter | Default | Effect of Increase |
|-----------|---------|-------------------|
| POLY_MARGIN_CURV | 120 | More robust curves, more noise |
| STALE_FIT_FRAMES | 12 | Lane memory (higher = longer) |
| estimated_lane_width | 280.0 | Affects target position |
| Dead reckon timeout (straight) | 5.0s | Recovery time on straights |
| Dead reckon timeout (curves) | 3.0s | Recovery time on curves |

### Priority 3 (Fine-Tuning)

| Parameter | Default | Note |
|-----------|---------|------|
| Yaw correction gain | 20.0 | Pixels per degree deviation |
| IMU yaw rate damping | 5.0 | Pixels per rad/s |
| Left sweep angle | -5.0° | Direction when lanes lost |

---

## Track Testing Guide

### Baseline Establishment

```python
# Run on reference track with good markings
# Collect metrics:
test_metrics = {
    'detection_rate': 0.95,      # Frames with sl or sr
    'dual_rate': 0.80,           # Frames with both lanes
    'confidence_mean': 0.92,     # Average confidence
    'target_x_std': 4.5,         # Stability (px)
    'mode_switches': 8,          # Transitions per 100 frames
}

# If any metric fails, adjust:
# - detection_rate < 0.85: Increase MIN_PIX_OK by 20-30
# - target_x_std > 8: Decrease EMA_ALPHA by 0.1
# - mode_switches > 15: Increase STALE_FIT_FRAMES by 3-5
```

### Curve Testing

```python
# Test on known curves (0.002-0.005 curvature)
# Verify:
# 1. y_eval increases (looking farther ahead)
# 2. No mode switching to SEARCH
# 3. target_x smooth transition (no jumps > 20px)
# 4. confidence stays > 0.8

# If failures:
# - Increase POLY_MARGIN_CURV (120 → 140)
# - Decrease MINPIX (50 → 30) for earlier detection
```

### Adverse Conditions

```python
# Shadow testing: Low-contrast areas
# Adjustment: Increase adaptive threshold constant

# Bright spots: Reflections/gloss
# Adjustment: Decrease morphology kernel size (5 → 3)

# Faded markings: Very low pixel count
# Adjustment: Decrease MIN_PIX_OK (200 → 150)
```

---

## Troubleshooting

### Lane Detection Fails (confidence drops to 0.0)

**Symptoms**: Vehicle loses lane guidance
**Diagnosis**:
1. Check binary image: Are lanes visible?
2. Is mode stuck in SEARCH?
3. How many frames lost?

**Solutions**:
1. Adjust preprocessing (CLAHE, threshold)
2. Reduce MIN_PIX_OK (200 → 150)
3. Increase SW_MARGIN (60 → 80)

### Steering Oscillates (wobble at 5-10 Hz)

**Symptoms**: Rapid left-right steering corrections
**Root Cause**: EMA alpha too high (no smoothing)

**Solutions**:
1. Decrease EMA_ALPHA (0.85 → 0.70)
2. Increase dead zone: ignore corrections < 3px
3. Check lookahead y_eval isn't too high

### Lane Width Estimate Jumps

**Symptoms**: `lane_width_px` varies 280±50 constantly
**Root Cause**: Dual-lane EMA weight too high

**Solutions**:
1. Increase EMA_ALPHA on width (0.8 → 0.95)
2. Add lane width sanity check (check outliers)
3. Use moving median instead of average

### False Positive Lanes (phantom lanes in noise)

**Symptoms**: Dual lanes detected but width invalid
**Root Cause**: `_width_sane()` thresholds too loose

**Solutions**:
1. Decrease WIDE_ROAD_PX (420 → 380)
2. Increase SINGLE_LANE_PX (200 → 230)
3. Increase MIN_PIX_OK (200 → 250)

---

## Summary: Quick Tuning Checklist

- [ ] Test baseline metrics on reference track
- [ ] Adjust MIN_PIX_OK if detection fails
- [ ] Adjust EMA_ALPHA if oscillation occurs
- [ ] Increase POLY_MARGIN_CURV for tight curves
- [ ] Verify dead reckoning timeout (3-5s)
- [ ] Test curve transitions (LEFT/RIGHT hints)
- [ ] Check phantom lane rejection (width sanity)
- [ ] Measure detection rate (should be > 85%)

