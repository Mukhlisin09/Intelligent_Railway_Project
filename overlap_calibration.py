from ultralytics import YOLO
import cv2
import numpy as np
import json
from shapely.geometry import Polygon, box

# =====================================================
# KONFIGURASI
# =====================================================
MODEL_PATH = "best v8.pt"
VIDEO_SOURCE = "perlintasan.mp4"

vehicle_labels = {
    "mobil",
    "motor",
    "truk",
    "bus",
    "sepeda",
    "kendaraan tradisional"
}

# =====================================================
# LOAD MODEL
# =====================================================
model = YOLO(MODEL_PATH)

# =====================================================
# LOAD ROI
# =====================================================
with open("roi_kendaraan.json", "r") as f:
    roi_points = json.load(f)["points"]

roi_polygon = Polygon(roi_points)

roi_np = np.array(roi_points, np.int32)

# =====================================================
# VIDEO
# =====================================================
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("ERROR kamera")
    exit()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    # gambar ROI
    cv2.polylines(display, [roi_np], True, (0,255,0), 2)

    results = model(frame, verbose=False)

    max_overlap = 0

    for result in results:

        boxes = result.boxes

        for box_data in boxes:

            cls_id = int(box_data.cls[0])
            label = model.names[cls_id]

            if label not in vehicle_labels:
                continue

            x1, y1, x2, y2 = map(int, box_data.xyxy[0])

            bbox_polygon = box(x1, y1, x2, y2)

            intersection_area = roi_polygon.intersection(bbox_polygon).area

            bbox_area = bbox_polygon.area

            overlap = 100 * intersection_area / bbox_area

            if overlap > max_overlap:
                max_overlap = overlap

            # bounding box
            cv2.rectangle(display,
                          (x1,y1),
                          (x2,y2),
                          (255,0,0),
                          2)

            text = f"{label} {overlap:.1f}%"

            cv2.putText(display,
                        text,
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255,0,0),
                        2)

    cv2.putText(display,
                f"MAX OVERLAP = {max_overlap:.1f} %",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3)

    cv2.putText(display,
                "Amati overlap saat kendaraan mulai masuk ROI",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,255),
                2)

    cv2.imshow("KALIBRASI THRESHOLD", display)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()