import cv2
import numpy as np
import time

# ==========================================
# VIDEO PATH
# ==========================================
video_path = "E:/BFMC/parking_romania.mp4"

cap = cv2.VideoCapture(video_path)

prev_time = time.time()

# ==========================================
# FUNCTION TO FIND INTERSECTION
# ==========================================
def line_intersection(line1, line2):

    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denominator = (x1 - x2) * (y3 - y4) - \
                  (y1 - y2) * (x3 - x4)

    if denominator == 0:
        return None

    px = ((x1*y2 - y1*x2) * (x3 - x4) -
          (x1 - x2) * (x3*y4 - y3*x4)) / denominator

    py = ((x1*y2 - y1*x2) * (y3 - y4) -
          (y1 - y2) * (x3*y4 - y3*x4)) / denominator

    return int(px), int(py)

# ==========================================
# MAIN LOOP
# ==========================================
while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        print("Video ended")
        break

    # ----------------------------------
    # RESIZE FRAME
    # ----------------------------------
    frame = cv2.resize(frame, (640, 480))

    height, width, _ = frame.shape

    center_x = width // 2

    # ----------------------------------
    # BOTTOM ROI
    # ----------------------------------
    roi_y = int(height * 0.60)

    roi = frame[roi_y:height, :]

    # ----------------------------------
    # GRAYSCALE
    # ----------------------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ----------------------------------
    # BLUR
    # ----------------------------------
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # ----------------------------------
    # CANNY
    # ----------------------------------
    edges = cv2.Canny(blur, 50, 150)

    # ----------------------------------
    # HOUGH LINES
    # ----------------------------------
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=80
    )

    horizontal_lines = []
    vertical_lines = []

    # ==========================================
    # DETECT LINES
    # ==========================================
    if lines is not None:

        for line in lines:

            x1, y1, x2, y2 = line[0]

            # Shift ROI
            y1 += roi_y
            y2 += roi_y

            # ----------------------------------
            # ANGLE
            # ----------------------------------
            angle = abs(
                np.degrees(
                    np.arctan2(y2 - y1, x2 - x1)
                )
            )

            # ----------------------------------
            # LENGTH
            # ----------------------------------
            line_length = np.sqrt(
                (x2 - x1) ** 2 +
                (y2 - y1) ** 2
            )

            # ==========================================
            # HORIZONTAL LINE
            # ==========================================
            if angle < 10:

                # remove tiny horizontal noise
                if line_length < 120:
                    continue

                horizontal_lines.append(
                    (x1, y1, x2, y2)
                )

                color = (255, 255, 0)

            # ==========================================
            # VERTICAL LINE (OLD LOGIC)
            # ==========================================
            else:

                vertical_lines.append(
                    (x1, y1, x2, y2)
                )

                color = (0, 255, 255)

            # ----------------------------------
            # DRAW LINE
            # ----------------------------------
            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                4
            )

    # ==========================================
    # INTERSECTION DETECTION
    # ==========================================
    slot_detected = False

    for h_line in horizontal_lines:

        for v_line in vertical_lines:

            point = line_intersection(
                h_line,
                v_line
            )

            if point is not None:

                px, py = point

                # inside frame
                if 0 <= px < width and 0 <= py < height:

                    slot_detected = True

                    # ----------------------------------
                    # RED INTERSECTION POINT
                    # ----------------------------------
                    cv2.circle(
                        frame,
                        (px, py),
                        8,
                        (0, 0, 255),
                        -1
                    )

    # ==========================================
    # SLOT DETECTED LABEL
    # ==========================================
    if slot_detected:

        cv2.putText(
            frame,
            "SLOT DETECTED",
            (180, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # ----------------------------------
    # CENTER LINE
    # ----------------------------------
    cv2.line(
        frame,
        (center_x, 0),
        (center_x, height),
        (0, 255, 255),
        3
    )

    # ----------------------------------
    # ROI LINE
    # ----------------------------------
    cv2.line(
        frame,
        (0, roi_y),
        (width, roi_y),
        (255, 255, 0),
        3
    )

    # ----------------------------------
    # FPS
    # ----------------------------------
    current_time = time.time()

    fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ----------------------------------
    # DISPLAY
    # ----------------------------------
    cv2.imshow(
        "Parking Slot Detector",
        frame
    )

    cv2.imshow(
        "Edges",
        edges
    )

    # ----------------------------------
    # EXIT
    # ----------------------------------
    key = cv2.waitKey(25)

    if key == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================
cap.release()

cv2.destroyAllWindows()