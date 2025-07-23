import cv2
import numpy as np
import os
import json
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

# Global variables
drawing = False
polygon_points = []
path_drawn = False
intrusion_log = []

# Create folder to store intruder images
os.makedirs("intrusions", exist_ok=True)

# Create or clear the JSON file initially
with open("intrusion_log.json", "w") as f:
    json.dump([], f, indent=4)

# Mouse callback function to draw polygon
def draw_polygon(event, x, y, flags, param):
    global drawing, polygon_points, path_drawn
    if event == cv2.EVENT_LBUTTONDOWN and not path_drawn:
        drawing = True
        polygon_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(polygon_points) > 2:
            path_drawn = True
            drawing = False

# Function to check if a point is inside the polygon
def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

# Setup window and mouse callback
cv2.namedWindow("PathGuard Zone Draw")
cv2.setMouseCallback("PathGuard Zone Draw", draw_polygon)

# Video input
cap = cv2.VideoCapture("cctv_footage.mp4")  # Replace with your video path

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    display_frame = frame.copy()
    frame_count += 1

    # Draw polygon and points
    if polygon_points:
        for point in polygon_points:
            cv2.circle(display_frame, point, 5, (0, 0, 255), -1)
        if path_drawn and len(polygon_points) > 2:
            cv2.polylines(display_frame, [np.array(polygon_points)], isClosed=True, color=(255, 0, 0), thickness=2)

    # Run YOLO only after polygon is drawn
    if path_drawn:
        results = model.predict(source=frame, show=False, conf=0.4)
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label == "person":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    if not point_in_polygon(center, polygon_points):
                        color = (0, 0, 255)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, "Out of Path!", (x1, y1 - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                        # Save frame image
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"intrusions/intrusion_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)

                        # Calculate video time
                        video_time_sec = frame_count / fps
                        h = int(video_time_sec // 3600)
                        m = int((video_time_sec % 3600) // 60)
                        s = video_time_sec % 60
                        time_str = f"{h:02}:{m:02}:{s:06.3f}"

                        # Create entry
                        entry = {
                            "image": os.path.basename(filename),
                            "frame": frame_count,
                            "video_time": time_str,
                            "timestamp": datetime.now().isoformat()
                        }

                        # Append to in-memory list
                        intrusion_log.append(entry)

                        # Live write to JSON file
                        with open("intrusion_log.json", "w") as f:
                            json.dump(intrusion_log, f, indent=4)

                    else:
                        color = (0, 255, 0)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, "In Safe Zone", (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display frame
    cv2.imshow("PathGuard Zone Draw", display_frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        polygon_points = []
        path_drawn = False
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
