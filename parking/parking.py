from .parking_config import PARKING_SPEED_MULTIPLIER, NORMAL_SPEED_MULTIPLIER
from .parking_imu import ParkingDistanceTracker
from .parking_detector import ParkingDetector
from .parking_slot_manager import ParkingSlotManager
from .parking_trajectory import ParkingTrajectory
from .parking_state_machine import ParkingStateMachine
from .parking_dashboard import ParkingDashboard

class ParkingSystem:
    def __init__(self, onnx_model_path="models/parking_car.onnx", 
                 left_csv="csv/left_parallel_parking.csv", 
                 right_csv="csv/right_parallel_parking.csv",
                 debug_dashboard=True):
        
        self.tracker = ParkingDistanceTracker()
        self.detector = ParkingDetector(onnx_model_path)
        self.slot_manager = ParkingSlotManager()
        self.trajectory = ParkingTrajectory(left_csv, right_csv)
        self.state_machine = ParkingStateMachine()
        self.last_debug_data = None

    def update(self, frame, dt, real_imu, reverse_parking_done=False, autonomous_mode=True, pedestrian_detected=False):
        """
        Master orchestration loop.
        Calls modules, manages flow, and outputs standard parking dict.
        """
        sign_detections = []
        detected_cars = []
        roi_frame = None
        debug_data = {}
        
        accel_forward = real_imu.get("accel_forward", 0.0) if isinstance(real_imu, dict) else real_imu
        
        if not autonomous_mode:
            self.tracker.reset()
            self.state_machine.reset()
            self.last_debug_data = None
            return {
                "parking_completed": False,
                "parking_failed": False,
                "selected_slot": None,
                "selected_side": None,
                "occupancy_status": {},
                "trajectory": None,
                "speed_multiplier": 1.0,
                "parking_mode_active": False,
                "parking_takeover": False
            }

        if pedestrian_detected:
            if self.state_machine.get_state() == 7:
                self.state_machine.reverse_parking_start_time += dt
            
            # Freeze parking logic but maintain output
            state_after = self.state_machine.get_state()
            traj_points = self.trajectory.trajectory_points if state_after >= 7 else None
            
            # Dashboard update logic
            if self.debug_dashboard:
                if not hasattr(self, 'dashboard'):
                    from .parking_dashboard import ParkingDashboard
                    self.dashboard = ParkingDashboard()
                    
                roi_frame = None
                if frame is not None and state_after >= 2:
                    h_img, w_img = frame.shape[:2]
                    roi_y_start = int(h_img * 0.75)
                    roi_frame = frame[roi_y_start:h_img, 0:w_img]
                    
                debug_data = {
                    "state": state_after,
                    "distance_cm": self.tracker.distance_cm,
                    "current_slot": self.slot_manager.current_slot,
                    "selected_slot": self.slot_manager.selected_slot,
                    "selected_side": self.slot_manager.selected_side,
                    "occupancy_map": self.slot_manager.occupancy_map,
                    "speed_multiplier": 0.0,
                    "parking_completed": self.state_machine.parking_completed,
                    "parking_failed": self.state_machine.parking_failed,
                    "trajectory": traj_points,
                    "roi_frame": roi_frame,
                    "full_frame": frame,
                    "sign_detections": sign_detections,
                    "car_detections": detected_cars,
                    "target_stop_distance": self.state_machine.target_stop_distance_cm,
                    "imu_data": {
                        "accel": accel_forward,
                        "velocity": self.tracker.velocity,
                        "distance_cm": self.tracker.distance_cm
                    },
                    "real_imu": real_imu if isinstance(real_imu, dict) else {}
                }
                self.last_debug_data = debug_data
                
            return {
                "parking_completed": self.state_machine.parking_completed,
                "parking_failed": self.state_machine.parking_failed,
                "selected_slot": self.slot_manager.selected_slot,
                "selected_side": self.slot_manager.selected_side,
                "occupancy_status": self.slot_manager.occupancy_map,
                "trajectory": traj_points,
                "speed_multiplier": 0.0,
                "parking_mode_active": state_after >= 1,
                "parking_takeover": False
            }

        # 1. Update IMU tracking
        self.tracker.update(accel_forward, dt)
        
        state_before = self.state_machine.get_state()
        
        # 2. Vision Processing
        sign_detected = False
        
        if state_before == 0:
            sign_detected, sign_detections = self.detector.detect_parking_sign(frame)
        elif state_before == 2:
            detected_cars = self.detector.detect_cars_in_roi(frame)
            
        # 3. Slot Management
        slot_crossed = False
        calculated_slot = self.slot_manager.current_slot
        
        if state_before == 2:
            slot_crossed, calculated_slot = self.slot_manager.update_slot_occupancy(
                self.tracker.distance_cm, 
                detected_cars
            )
            if slot_crossed:
                self.slot_manager.decide_target_slot()
                self.slot_manager.update_current_slot(calculated_slot)

        # 4. State Machine Transition
        self.state_machine.transition(
            sign_detected=sign_detected,
            slot_crossed=slot_crossed,
            selected_slot=self.slot_manager.selected_slot,
            calculated_slot=calculated_slot,
            current_distance_cm=self.tracker.distance_cm,
            reverse_parking_done=reverse_parking_done
        )
        
        state_after = self.state_machine.get_state()

        # Handle post-transition specific actions (one-shot actions)
        if state_before == 0 and state_after == 1:
            self.tracker.reset()
            
        elif state_before == 5 and state_after == 6:
            if self.slot_manager.selected_side == "left":
                self.trajectory.load_left_trajectory()
            else:
                self.trajectory.load_right_trajectory()

        # 5. Output Formatting
        speed_multiplier = NORMAL_SPEED_MULTIPLIER if state_after == 0 else PARKING_SPEED_MULTIPLIER
        traj_points = self.trajectory.trajectory_points if state_after >= 7 else None
        
        if self.debug_dashboard:
            if not hasattr(self, 'dashboard'):
                from .parking_dashboard import ParkingDashboard
                self.dashboard = ParkingDashboard()
                
            # We need roi_frame if state >= 2
            roi_frame = None
            if frame is not None and state_after >= 2:
                h_img, w_img = frame.shape[:2]
                roi_y_start = int(h_img * 0.75)
                roi_frame = frame[roi_y_start:h_img, 0:w_img]
                
            debug_data = {
                "state": state_after,
                "distance_cm": self.tracker.distance_cm,
                "current_slot": self.slot_manager.current_slot,
                "selected_slot": self.slot_manager.selected_slot,
                "selected_side": self.slot_manager.selected_side,
                "occupancy_map": self.slot_manager.occupancy_map,
                "speed_multiplier": speed_multiplier,
                "parking_completed": self.state_machine.parking_completed,
                "parking_failed": self.state_machine.parking_failed,
                "trajectory": traj_points,
                "roi_frame": roi_frame,
                "full_frame": frame,
                "sign_detections": sign_detections,
                "car_detections": detected_cars,
                "target_stop_distance": self.state_machine.target_stop_distance_cm,
                "imu_data": {
                    "accel": accel_forward,
                    "velocity": self.tracker.velocity,
                    "distance_cm": self.tracker.distance_cm
                },
                "real_imu": real_imu if isinstance(real_imu, dict) else {}
            }
            self.last_debug_data = debug_data

        return {
            "parking_completed": self.state_machine.parking_completed,
            "parking_failed": self.state_machine.parking_failed,
            "selected_slot": self.slot_manager.selected_slot,
            "selected_side": self.slot_manager.selected_side,
            "occupancy_status": self.slot_manager.occupancy_map,
            "trajectory": traj_points,
            "speed_multiplier": speed_multiplier,
            "parking_mode_active": state_after >= 1,
            "parking_takeover": 5 <= state_after <= 7
        }

    def render_dashboard(self, frame):
        try:
            if not hasattr(self, 'dashboard'):
                from .parking_dashboard import ParkingDashboard
                self.dashboard = ParkingDashboard()

            data = getattr(self, 'last_debug_data', None)
            if data is None:
                data = {
                    "state": -1,
                    "distance_cm": 0.0,
                    "current_slot": 0,
                    "selected_slot": None,
                    "selected_side": None,
                    "occupancy_map": {},
                    "speed_multiplier": 1.0,
                    "parking_completed": False,
                    "parking_failed": False,
                    "trajectory": None,
                    "roi_frame": None,
                    "full_frame": frame,
                    "sign_detections": [],
                    "car_detections": [],
                    "target_stop_distance": 0.0,
                    "imu_data": {"accel": 0.0, "velocity": 0.0, "distance_cm": 0.0},
                    "real_imu": {},
                    "model_loaded": getattr(self.detector, 'parking_detection_enabled', True)
                }
            else:
                data = data.copy()
                data["full_frame"] = frame
                data["model_loaded"] = getattr(self.detector, 'parking_detection_enabled', True)

            self.dashboard.update(data)
            # print("[Parking Dashboard] Rendered") # Can be noisy, so leaving commented out or just let dashboard run
        except Exception as e:
            print(f"[Parking Dashboard Error] {e}")
