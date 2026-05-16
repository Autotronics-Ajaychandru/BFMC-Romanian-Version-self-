import cv2
import time
import sys
import os
import numpy as np

# Add the parent directory to the path so we can import perception
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from perception.lane_detector import LaneDetector
except ImportError as e:
    print(f"Error importing LaneDetector: {e}")
    sys.exit(1)

def simulate_video(video_path):
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    detector = LaneDetector()
    
    # Pre-calculate FPS for real-time simulation
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0
    delay_ms = int(1000 / fps)

    print("Starting simulation... Press 'q' to quit.")
    
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break

        # Record processing time
        t1 = time.time()
        
        # Process the frame through the Lane Detector
        # We simulate a constant current_yaw of 0.0 for testing
        result = detector.process(frame, dt=1.0/fps, current_yaw=0.0)
        
        t2 = time.time()
        process_time_ms = (t2 - t1) * 1000

        # Create a dashboard overlay
        overlay = frame.copy()
        
        # Draw status info
        cv2.putText(overlay, f"Proc Time: {process_time_ms:.1f}ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"Steer Target: {result.target_x:.1f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"Confidence: {result.confidence:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(overlay, f"Anchor: {result.anchor}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the debug views
        cv2.imshow("Main Feed (Result)", overlay)
        
        if result.lane_dbg is not None:
            cv2.imshow("Lane Fitting Debug", result.lane_dbg)
        
        if result.warped_binary is not None:
            # Show the binary image to check thresholding quality
            cv2.imshow("Binary (Perspective)", result.warped_binary)

        frame_count += 1

        # Wait for keypress and regulate frame rate
        key = cv2.waitKey(max(1, delay_ms - int(process_time_ms))) & 0xFF
        if key == ord('q'):
            print("User interrupted the simulation.")
            break

    total_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds.")
    print(f"Average FPS: {frame_count / total_time:.2f}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path provided by user
    video_path = r"E:\BFMC-Romanian-Version\BFMC_2026-main1\video_20260516_132240.avi"
    simulate_video(video_path)
