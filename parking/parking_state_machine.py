import time
from .parking_config import SIGN_CONFIRM_FRAMES, MAX_SLOTS, SLOT_LENGTH_CM, REVERSE_TIMEOUT

class ParkingStateMachine:
    def __init__(self):
        self.state = 0
        self.sign_detection_count = 0
        self.target_stop_distance_cm = 0.0
        self.reverse_parking_start_time = 0.0
        self.parking_completed = False
        self.parking_failed = False

    def get_state(self):
        return self.state

    def reset(self):
        self.state = 0
        self.sign_detection_count = 0
        self.target_stop_distance_cm = 0.0
        self.reverse_parking_start_time = 0.0
        self.parking_completed = False
        self.parking_failed = False

    def transition(self, 
                   sign_detected=False, 
                   slot_crossed=False, 
                   selected_slot=None, 
                   calculated_slot=1,
                   current_distance_cm=0.0,
                   reverse_parking_done=False):
        """
        Evaluates and transitions to the next state based on inputs.
        """
        # State 0: NORMAL DRIVING
        if self.state == 0:
            if sign_detected:
                self.sign_detection_count += 1
                if self.sign_detection_count >= SIGN_CONFIRM_FRAMES:
                    print("[Parking] Sign detected")
                    self.state = 1
            else:
                self.sign_detection_count = 0

        # State 1: PARKING SIGN DETECTED
        elif self.state == 1:
            print("[Parking] ROI activated")
            self.state = 2

        # State 2: SLOT SCANNING
        elif self.state == 2:
            if slot_crossed and selected_slot is not None:
                self.state = 3
                
            elif calculated_slot > MAX_SLOTS:
                print("[Parking] No free slots found.")
                self.parking_failed = True
                self.state = 8

        # State 3: TARGET SLOT DECISION
        elif self.state == 3:
            target_stop_slot = min(selected_slot + 1, MAX_SLOTS)
            self.target_stop_distance_cm = target_stop_slot * SLOT_LENGTH_CM
            print(f"[Parking] Moving to {self.target_stop_distance_cm} cm")
            self.state = 4

        # State 4: MOVE TO TARGET DISTANCE
        elif self.state == 4:
            if current_distance_cm >= self.target_stop_distance_cm:
                self.state = 5

        # State 5: STOP VEHICLE
        elif self.state == 5:
            self.state = 6

        # State 6: LOAD CSV TRAJECTORY
        elif self.state == 6:
            print("[Parking] CSV loaded")
            print("[Parking] Reverse parking started")
            self.reverse_parking_start_time = time.time()
            self.state = 7

        # State 7: REVERSE PARKING EXECUTION
        elif self.state == 7:
            if reverse_parking_done:
                self.state = 8
            elif time.time() - self.reverse_parking_start_time > REVERSE_TIMEOUT:
                print("[Parking] Reverse parking timeout")
                self.parking_failed = True
                self.state = 8

        # State 8: PARKING COMPLETE
        elif self.state == 8:
            if not self.parking_failed:
                self.parking_completed = True
