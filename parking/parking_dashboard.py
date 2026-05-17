import time
import numpy as np
try:
    import cv2
except ImportError:
    pass

class ParkingDashboard:
    def __init__(self):
        self.canvas_w = 1280
        self.canvas_h = 720
        self.window_name = "BFMC Parking Dashboard"
        self.update_every_n_frames = 2
        self.frame_count = 0
        
        # Colors (BGR for OpenCV)
        self.C_GREEN = (0, 255, 0)
        self.C_RED = (0, 0, 255)
        self.C_BLUE = (255, 0, 0)
        self.C_YELLOW = (0, 255, 255)
        self.C_CYAN = (255, 255, 0)
        self.C_PURPLE = (255, 0, 255)
        self.C_WHITE = (255, 255, 255)
        self.C_BLACK = (0, 0, 0)
        self.C_GRAY = (50, 50, 50)
        
        self.state_names = {
            0: "NORMAL DRIVING",
            1: "PARKING SIGN DETECTED",
            2: "SLOT SCANNING",
            3: "TARGET SLOT DECISION",
            4: "MOVE TO TARGET DISTANCE",
            5: "STOP VEHICLE",
            6: "LOAD CSV TRAJECTORY",
            7: "REVERSE PARKING EXECUTION",
            8: "PARKING COMPLETE"
        }

        self.last_time = time.time()

    def update(self, debug_data):
        self.frame_count += 1
        if self.frame_count % self.update_every_n_frames != 0:
            return

        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time

        # --- 1. Main Camera Feed (Top Left 640x360) ---
        frame = debug_data.get("full_frame")
        if frame is not None:
            # draw bounding boxes
            frame_disp = frame.copy()
            # The coordinates are typically relative to original shape or 640x640.
            # Assuming cx, cy, w, h are based on 640x640 inference shape, we need to map them.
            # We'll just draw them if they exist in detections list
            h_f, w_f = frame_disp.shape[:2]
            
            # Sign detections
            for d in debug_data.get("sign_detections", []):
                cx, cy, bw, bh = d.get('cx',0), d.get('cy',0), d.get('w',0), d.get('h',0)
                # Map from model shape to original frame shape? 
                # If they are already in pixel coordinates of the resized model, we should scale.
                # Since parking_detector doesn't scale them back, we assume they are in 640x640 space.
                scale_x = w_f / 640.0
                scale_y = h_f / 640.0
                x1 = int((cx - bw/2) * scale_x)
                y1 = int((cy - bh/2) * scale_y)
                x2 = int((cx + bw/2) * scale_x)
                y2 = int((cy + bh/2) * scale_y)
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), self.C_PURPLE, 2)
                cv2.putText(frame_disp, f"SIGN {d.get('confidence', 0):.2f}", (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_PURPLE, 1)

            # Resize to fit 640x360
            frame_resized = cv2.resize(frame_disp, (640, 360))
            
            # Overlay State Text
            st_idx = debug_data.get("state", 0)
            
            if not debug_data.get("model_loaded", True):
                cv2.putText(frame_resized, f"Parking Model: NOT LOADED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_RED, 2)
                cv2.putText(frame_resized, f"Parking State: IDLE", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_YELLOW, 2)
            elif st_idx == -1:
                cv2.putText(frame_resized, f"Parking State: IDLE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_YELLOW, 2)
            else:
                cv2.putText(frame_resized, f"State: {self.state_names.get(st_idx, '')}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_CYAN, 2)
                cv2.putText(frame_resized, f"Speed: x{debug_data.get('speed_multiplier', 1.0):.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_CYAN, 2)
                cv2.putText(frame_resized, f"Dist: {debug_data.get('distance_cm', 0):.1f} cm", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_CYAN, 2)
            
            canvas[0:360, 0:640] = frame_resized

        # FPS overlay
        cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_GREEN, 2)

        # --- 2. Parking ROI (Top Right 640x360) ---
        roi_frame = debug_data.get("roi_frame")
        if roi_frame is not None:
            roi_disp = roi_frame.copy()
            rh, rw = roi_disp.shape[:2]
            
            # Draw center line
            cv2.line(roi_disp, (rw//2, 0), (rw//2, rh), self.C_CYAN, 2)
            cv2.putText(roi_disp, "LEFT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.C_GREEN, 2)
            cv2.putText(roi_disp, "RIGHT", (rw//2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.C_BLUE, 2)

            # Draw car bounding boxes
            for car in debug_data.get("car_detections", []):
                # cx, cy, w, h are in 640x640 space relative to the ROI image resize
                bbox = car.get('bbox', (0,0,0,0))
                cx, cy, bw, bh = bbox
                scale_x = rw / 640.0
                scale_y = rh / 640.0
                x1 = int((cx - bw/2) * scale_x)
                y1 = int((cy - bh/2) * scale_y)
                x2 = int((cx + bw/2) * scale_x)
                y2 = int((cy + bh/2) * scale_y)
                color = self.C_GREEN if car.get('side') == 'left' else self.C_BLUE
                cv2.rectangle(roi_disp, (x1, y1), (x2, y2), color, 2)
                cv2.putText(roi_disp, f"CAR {car.get('conf', 0):.2f}", (x1, max(y1-10, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            roi_resized = cv2.resize(roi_disp, (640, 360))
            canvas[0:360, 640:1280] = roi_resized
            cv2.rectangle(canvas, (640, 0), (1280, 360), self.C_CYAN, 2) # ROI border

        # --- 3. Parking Status (Bottom Left 320x360) ---
        cv2.rectangle(canvas, (0, 360), (320, 720), self.C_GRAY, -1)
        cv2.rectangle(canvas, (0, 360), (320, 720), self.C_WHITE, 1)
        
        y_offset = 380
        st = debug_data.get("state", 0)
        cv2.putText(canvas, f"PARKING STATUS", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.C_WHITE, 2)
        
        mode_text = "ACTIVE" if st >= 1 else ("IDLE" if st == -1 else "INACTIVE")
        armed_text = "YES" if st >= 1 and st < 5 else "NO"
        takeover_text = "YES" if st >= 5 and st <= 7 else "NO"
        sign_text = "DETECTED" if st >= 1 else "NOT DETECTED"
        model_text = "LOADED" if debug_data.get("model_loaded", True) else "NOT LOADED"
        
        cv2.putText(canvas, f"Parking Mode: {mode_text}", (10, y_offset+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_GREEN if st>=1 else self.C_YELLOW, 2)
        cv2.putText(canvas, f"Parking Armed: {armed_text}", (10, y_offset+80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_PURPLE if armed_text=="YES" else self.C_WHITE, 1)
        cv2.putText(canvas, f"Parking Takeover: {takeover_text}", (10, y_offset+120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_RED if takeover_text=="YES" else self.C_WHITE, 2)
        cv2.putText(canvas, f"Parking Sign: {sign_text}", (10, y_offset+160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_CYAN, 1)
        cv2.putText(canvas, f"Model: {model_text}", (10, y_offset+200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_GREEN if debug_data.get("model_loaded", True) else self.C_RED, 1)
        
        cv2.putText(canvas, f"State {st}: {self.state_names.get(st, 'IDLE')}", (10, y_offset+240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        cv2.putText(canvas, f"Target Stop: {debug_data.get('target_stop_distance', 0)} cm", (10, y_offset+280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_YELLOW, 1)

        # Control Hierarchy Status
        cv2.rectangle(canvas, (0, 320), (320, 360), (40, 20, 20) if takeover_text == "YES" else (20, 40, 20), -1)
        if takeover_text == "YES":
            cv2.putText(canvas, "Parking TAKEOVER ACTIVE", (10, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_RED, 2)
        else:
            cv2.putText(canvas, "Lane Following ACTIVE | Parking BACKGROUND", (10, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_GREEN, 1)

        # --- 4. IMU Debug (Bottom Middle 320x360) ---
        cv2.rectangle(canvas, (320, 360), (640, 720), (30, 30, 30), -1)
        cv2.rectangle(canvas, (320, 360), (640, 720), self.C_WHITE, 1)
        
        imu = debug_data.get("imu_data", {})
        real_imu = debug_data.get("real_imu", {})
        
        cv2.putText(canvas, f"REAL IMU", (330, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_CYAN, 2)
        cv2.putText(canvas, f"Accel X: {real_imu.get('accel_x', 0):.3f}", (330, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        cv2.putText(canvas, f"Accel Y: {real_imu.get('accel_y', 0):.3f}", (330, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        cv2.putText(canvas, f"Accel Z: {real_imu.get('accel_forward', 0):.3f}", (330, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        
        cv2.putText(canvas, f"Gyro X: {real_imu.get('gyro_x', 0):.3f}", (330, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        cv2.putText(canvas, f"Gyro Y: {real_imu.get('gyro_y', 0):.3f}", (330, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        cv2.putText(canvas, f"Gyro Z: {real_imu.get('gyro_z', 0):.3f}", (330, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        
        cv2.putText(canvas, f"Yaw:   {real_imu.get('yaw', 0):.1f}", (330, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        cv2.putText(canvas, f"Pitch: {real_imu.get('pitch', 0):.1f}", (330, 570), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        cv2.putText(canvas, f"Roll:  {real_imu.get('roll', 0):.1f}", (330, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)

        cv2.line(canvas, (470, 360), (470, 720), (100,100,100), 1)

        cv2.putText(canvas, f"PARKING RESET IMU", (480, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_YELLOW, 2)
        cv2.putText(canvas, f"Distance: {imu.get('distance_cm', 0):.1f} cm", (480, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_GREEN, 2)
        cv2.putText(canvas, f"Velocity: {imu.get('velocity', 0):.4f}", (480, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        
        cv2.putText(canvas, f"Fwd Accel: {imu.get('accel', 0):.4f}", (480, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        cv2.putText(canvas, f"Reset Dist: 0.0", (480, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
        cv2.putText(canvas, f"(Db: 0.05, Damp: 0.98)", (480, 690), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.C_GRAY, 1)

        # --- 5. Slot Occupancy Map (Bottom Right-TopLeft 320x180) ---
        cv2.rectangle(canvas, (640, 360), (960, 540), (20, 20, 20), -1)
        cv2.rectangle(canvas, (640, 360), (960, 540), self.C_WHITE, 1)
        
        cv2.putText(canvas, f"SLOT MAP & DETECTION", (650, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_WHITE, 2)
        
        sign_dets = debug_data.get("sign_detections", [])
        sign_conf = sign_dets[0].get("confidence", 0.0) if sign_dets else 0.0
        cv2.putText(canvas, f"Parking Sign: {'DETECTED' if sign_dets else 'NONE'}", (650, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_GREEN if sign_dets else self.C_WHITE, 1)
        cv2.putText(canvas, f"Confidence: {sign_conf:.2f}", (800, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_CYAN if sign_dets else self.C_WHITE, 1)
        
        left_car = any(c.get('side') == 'left' for c in debug_data.get("car_detections", []))
        right_car = any(c.get('side') == 'right' for c in debug_data.get("car_detections", []))
        
        cv2.putText(canvas, f"Left Car: {'YES' if left_car else 'NO'}", (650, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_GREEN if left_car else self.C_WHITE, 1)
        cv2.putText(canvas, f"Right Car: {'YES' if right_car else 'NO'}", (800, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.C_BLUE if right_car else self.C_WHITE, 1)
        
        occ = debug_data.get("occupancy_map", {})
        for i in range(1, 4): # Display first 3 slots to save space
            slot_info = occ.get(i, {'left': False, 'right': False})
            l_char = "X" if slot_info['left'] else "_"
            r_char = "X" if slot_info['right'] else "_"
            cv2.putText(canvas, f"L{i} [{l_char}]", (650, 450 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_RED if slot_info['left'] else self.C_GREEN, 1)
            cv2.putText(canvas, f"R{i} [{r_char}]", (800, 450 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_RED if slot_info['right'] else self.C_GREEN, 1)

        # --- 6. Parking Decision Engine (Bottom Right-BotLeft 320x180) ---
        cv2.rectangle(canvas, (640, 540), (960, 720), (40, 40, 40), -1)
        cv2.rectangle(canvas, (640, 540), (960, 720), self.C_WHITE, 1)
        
        cv2.putText(canvas, f"DECISION ENGINE", (650, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_WHITE, 2)
        cv2.putText(canvas, f"Curr Slot: {debug_data.get('current_slot', 1)}", (650, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_WHITE, 1)
        
        sel_slot = debug_data.get("selected_slot")
        sel_side = debug_data.get("selected_side")
        if sel_slot is not None:
            cv2.putText(canvas, f"Selected: Slot {sel_slot}", (650, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_GREEN, 2)
            cv2.putText(canvas, f"Side: {str(sel_side).upper()}", (650, 650), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_BLUE, 2)
        else:
            cv2.putText(canvas, f"Selected: NONE", (650, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_YELLOW, 2)
            
        cv2.putText(canvas, f"Completed: {'YES' if debug_data.get('parking_completed') else 'NO'}", (800, 590), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_GREEN if debug_data.get('parking_completed') else self.C_WHITE, 1)
        cv2.putText(canvas, f"Failed: {'YES' if debug_data.get('parking_failed') else 'NO'}", (800, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_RED if debug_data.get('parking_failed') else self.C_WHITE, 1)

        # --- 7. Reverse Parking Trajectory (Bottom Right-Right 320x360) ---
        cv2.rectangle(canvas, (960, 360), (1280, 720), (10, 10, 10), -1)
        cv2.rectangle(canvas, (960, 360), (1280, 720), self.C_WHITE, 1)
        
        cv2.putText(canvas, f"TRAJECTORY", (970, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_WHITE, 2)
        traj = debug_data.get("trajectory")
        if traj:
            cv2.putText(canvas, f"Loaded Points: {len(traj)}", (970, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_GREEN, 1)
            cv2.putText(canvas, f"Time  | Steer | Speed", (970, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_CYAN, 1)
            # Display first few points
            for i in range(min(8, len(traj))):
                pt = traj[i]
                cv2.putText(canvas, f"{pt['time']:.1f} | {pt['steering']:.1f} | {pt['speed']:.2f}", 
                            (970, 465 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_WHITE, 1)
            if len(traj) > 8:
                cv2.putText(canvas, f"... and {len(traj)-8} more", (970, 465 + 8*25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.C_GRAY, 1)
        else:
            cv2.putText(canvas, f"No Trajectory Loaded", (970, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.C_YELLOW, 1)

        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)
