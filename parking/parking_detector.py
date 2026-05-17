# pyrefly: ignore [missing-import]
import numpy as np
from .parking_config import CONF_THRESHOLD

try:
    import cv2
except ImportError:
    pass

try:
    # pyrefly: ignore [missing-import]
    import onnxruntime as ort
except ImportError:
    pass

class ParkingDetector:
    def __init__(self, onnx_model_path="models/parking_car.onnx"):
        self.session = None
        self.parking_detection_enabled = True
        try:
            self.session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            print(f"[Parking] Model not found \u2014 detection disabled: {e}")
            self.parking_detection_enabled = False

    def _preprocess_frame(self, frame):
        if frame is None:
            return None
        try:
            input_shape = self.session.get_inputs()[0].shape
            h = input_shape[2] if isinstance(input_shape[2], int) else 640
            w = input_shape[3] if isinstance(input_shape[3], int) else 640
        except:
            h, w = 640, 640
        
        resized = cv2.resize(frame, (w, h))
        img = resized.transpose((2, 0, 1)) # HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)
        img /= 255.0
        return img

    def _detect(self, frame):
        if self.session is None or not self.parking_detection_enabled or frame is None:
            return []
            
        img = self._preprocess_frame(frame)
        if img is None:
            return []
            
        try:
            outputs = self.session.run(None, {self.input_name: img})
        except Exception as e:
            return []
        
        detections = []
        out = outputs[0]
        
        if len(out.shape) == 3:
            if out.shape[1] < out.shape[2]:
                out = out[0].T
            else:
                out = out[0]
                
            for row in out:
                class_scores = row[4:]
                if len(class_scores) == 0:
                    continue
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                
                if confidence > CONF_THRESHOLD:
                    cx, cy, w, h = row[0:4]
                    detections.append({
                        "class_id": class_id + 1,
                        "confidence": confidence,
                        "cx": cx, "cy": cy, "w": w, "h": h
                    })
        return detections

    def detect_parking_sign(self, frame):
        """Returns True and detections if a parking sign (Class 1) is detected."""
        detections = self._detect(frame)
        has_sign = any(d["class_id"] == 1 for d in detections)
        return has_sign, detections

    def detect_cars_in_roi(self, frame):
        """
        Extracts bottom quarter ROI, detects cars (Class 2), 
        and splits them into left/right side.
        Returns: list of dicts with 'side' ('left' or 'right')
        """
        if frame is None:
            return []
            
        h_img, w_img = frame.shape[:2]
        roi_y_start = int(h_img * 0.75)
        roi = frame[roi_y_start:h_img, 0:w_img]
        
        detections = self._detect(roi)
        
        roi_width = roi.shape[1]
        mid_x = roi_width / 2.0
        
        cars = []
        for det in detections:
            if det["class_id"] == 2:
                is_left = det["cx"] < mid_x
                cars.append({
                    'side': 'left' if is_left else 'right',
                    'conf': det.get('confidence', 0),
                    'bbox': (det['cx'], det['cy'], det['w'], det['h'])
                })
        return cars
