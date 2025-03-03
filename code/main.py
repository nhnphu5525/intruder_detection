import cv2
import numpy as np
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load model
model = YOLO("HumanDetection.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=10)

# 3D Intrusion Zone (choosed by user)
intrusion_zone_3D = {}
current_zone = []
zone_id = 1
ZONE_HEIGHT = 180  # Height of the zone in cm

# Store intrusion logs
intrusion_timers = {}
intrusion_logs = {"zone_1": {}, "zone_2": {}}
INTRUSION_THRESHOLD = 3  # Time in seconds

# Video path
vid_path = "mall_test.mp4"

def mouse_callback(event, x, y, flags, param):
    global current_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        current_zone.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and current_zone:
        current_zone.pop()

def create_intrusion_zone():
    global zone_id, intrusion_zone_3D, current_zone
    if len(current_zone) < 3:
        print("Please choose at least 3 points.")
        return
    bottom = np.array([[x, y, 0] for x, y in current_zone], np.float32)
    top = np.array([[x, y, ZONE_HEIGHT] for x, y in current_zone], np.float32)
    
    opening_time = input(f"Enter opening time for zone_{zone_id} (HH:MM:SS): ")
    closing_time = input(f"Enter closing time for zone_{zone_id} (HH:MM:SS): ")
    
    intrusion_zone_3D[f"zone_{zone_id}"] = {
        "bottom": bottom,
        "top": top,
        "closing_time": closing_time,
        "opening_time": opening_time
    }
    zone_id += 1
    current_zone = []
    print(f"Successfully created intrusion zone_{zone_id - 1} with open time {opening_time} - {closing_time}")

def is_person_inside_intrusion_zone(bbox, zone):
    x_min, y_min, x_max, y_max = bbox
    foot_x = (x_min + x_max) // 2
    foot_y = y_max
    height = (y_max - y_min) * 1.68 / 150  # Estimate the height of the person
    foot_point = np.array([foot_x, foot_y, 0], np.float32)
    head_point = np.array([foot_x, foot_y, height], np.float32)
    
    bottom, top = zone["bottom"], zone["top"]
    
    # Check if the foot point is inside the bottom polygon
    if cv2.pointPolygonTest(bottom[:, :2].astype(np.int32), (foot_x, foot_y), False) >= 0:
        min_z, max_z = np.min(bottom[:, 2]), np.max(top[:, 2])
        return min_z <= head_point[2] <= max_z
    return False

def is_outside_business_hours(zone_name):
    current_time = time.strftime("%H:%M:%S")
    opening_time = intrusion_zone_3D[zone_name]["opening_time"]
    closing_time = intrusion_zone_3D[zone_name]["closing_time"]

    # In case the closing time is on the next day
    if closing_time < opening_time:
        if current_time >= opening_time:
            return False  # In opening time
        if current_time < closing_time:
            return True   # Out of closing time

    # In case the closing time is on the same day
    return current_time < opening_time or current_time >= closing_time

# Create intrusion zone 

cap = cv2.VideoCapture(vid_path)
cv2.namedWindow("Select Intrusion Zone")
cv2.setMouseCallback("Select Intrusion Zone", mouse_callback)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    for point in current_zone:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
    if len(current_zone) > 1:
        cv2.polylines(frame, [np.array(current_zone, np.int32)], isClosed=True, color=(0, 255, 255), thickness=2)
    
    for zone_name, zone in intrusion_zone_3D.items():
        text = f"{zone_name}: {zone['opening_time']} - {zone['closing_time']}"
        cv2.putText(frame, text, (20, 30 * (int(zone_name.split('_')[1]) + 1)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow("Select Intrusion Zone", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        create_intrusion_zone()

cv2.destroyAllWindows()
cap.release()
print("Vùng intrusion đã được tạo: ", intrusion_zone_3D)

# Open camera or video file
#cap = cv2.VideoCapture(0)  # Replace video file if needed: cap = cv2.VideoCapture("test.mp4")
cap = cv2.VideoCapture(vid_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Draw intrusion zones
    for zone in intrusion_zone_3D.values():
        bottom = zone["bottom"][:, :2].astype(np.int32)
        top = zone["top"][:, :2].astype(np.int32) + np.array([0, -200])  # Move the top plane up
        
        cv2.polylines(frame, [bottom.reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.polylines(frame, [top.reshape((-1, 1, 2))], isClosed=True, color=(0, 0, 255), thickness=2)

        closing_time_text = f"Open time: {zone['opening_time']} - {zone['closing_time']}"
        text_position = (bottom[0][0], bottom[0][1] - 10)
        cv2.putText(frame, closing_time_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        for b, t in zip(bottom, top):
            cv2.line(frame, tuple(b), tuple(t), (0, 255, 255), 2)  # draw lines between bottom and top
    
    # Detect people using YOLO
    results = model(frame)
    detections = []
    
    for result in results:
        for box in result.boxes:
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])

            if class_id == 0 and confidence > 0.5:
                detections.append(([x_min, y_min, x_max - x_min, y_max - y_min], confidence))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        x_min, y_min, x_max, y_max = map(int, ltrb)
        
        for zone_name, zone in intrusion_zone_3D.items():
            if is_outside_business_hours(zone_name) and is_person_inside_intrusion_zone((x_min, y_min, x_max, y_max), zone):
                if track_id not in intrusion_timers:
                    intrusion_timers[track_id] = {"zone_1": None, "zone_2": None}
                
                if intrusion_timers[track_id][zone_name] is None:
                    intrusion_timers[track_id][zone_name] = time.time()
                    intrusion_logs[zone_name].setdefault(track_id, []).append(time.strftime("%Y-%m-%d %H:%M:%S"))
                
                elapsed_time = time.time() - intrusion_timers[track_id][zone_name]
                if elapsed_time >= INTRUSION_THRESHOLD:
                    color = (0, 0, 255)
                    text = f"Intruder {track_id} ({int(elapsed_time)}s) in {zone_name}"
                else:
                    color = (0, 255, 255)
                    text = f"Warning {track_id} ({int(elapsed_time)}s) in {zone_name}"
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                if track_id in intrusion_timers and intrusion_timers[track_id][zone_name] is not None:
                    intrusion_timers[track_id][zone_name] = None
    
    cv2.imshow("Intrusion Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Print intrusion logs
print("\nIntrusion Log:")
for zone, logs in intrusion_logs.items():
    print(f"\n {zone.upper()}:")
    for person_id, timestamps in logs.items():
        print(f" - Person ID {person_id}: {timestamps}")