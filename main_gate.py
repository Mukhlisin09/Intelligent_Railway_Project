# ==========================================================
# IMPORT LIBRARY
# ==========================================================
from ultralytics import YOLO
import cv2
import numpy as np
import json
import serial
import time
from shapely.geometry import Polygon, box
import torch

# ==========================================================
# KONFIGURASI
# ==========================================================
MODEL_PATH = "best v8.pt"
VIDEO_SOURCE = "perlintasan.mp4"

CONF_THRESH = 0.35

ROI_FILE = "roi_kendaraan.json"
THRESHOLD_FILE = "threshold.json"

OUTPUT_FILE = "hasil_deteksi_2.mp4"

SERIAL_PORT = "COM5"
BAUDRATE = 115200

TRAIN_LOST_FRAME = 20

vehicle_labels = {
    "mobil",
    "motor",
    "truk",
    "bus",
    "sepeda",
    "kendaraan tradisional"
}

train_label = "kereta"

# ==========================================================
# KONEKSI SERIAL STM32
# ==========================================================
try:
    ser = serial.Serial(
        SERIAL_PORT,
        BAUDRATE,
        timeout=1
    )
    time.sleep(2)
    print("Serial connected")
except Exception as e:
    print("Serial error:", e)
    ser = None

# ==========================================================
# LOAD MODEL
# ==========================================================
model = YOLO(MODEL_PATH)

# ==========================================================
# LOAD ROI
# ==========================================================
with open(ROI_FILE, "r") as f:
    roi_points = json.load(f)["points"]

roi_polygon = Polygon(roi_points)
roi_np = np.array(roi_points, np.int32)

ROI_TRAIN_FILE = "roi_kereta.json"

with open(ROI_TRAIN_FILE, "r") as f:
    roi_train_points = json.load(f)["points"]

roi_train_polygon = Polygon(roi_train_points)
roi_train_np = np.array(roi_train_points, np.int32)

# ==========================================================
# LOAD THRESHOLD KENDARAAN
# ==========================================================
with open(THRESHOLD_FILE, "r") as f:
    OVERLAP_THRESHOLD = json.load(f)["overlap_threshold"]

print("Threshold =", OVERLAP_THRESHOLD)

# ==========================================================
# VIDEO CAPTURE
# ==========================================================
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("ERROR membuka kamera")
    exit()

# Ambil ukuran frame
ret, frame = cap.read()

if not ret:
    print("ERROR membaca frame")
    exit()

frame_height, frame_width = frame.shape[:2]

# ==========================================================
# VIDEO WRITER
# ==========================================================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

out = cv2.VideoWriter(
    OUTPUT_FILE,
    fourcc,
    20,
    (frame_width, frame_height)
)

# ==========================================================
# STATE MACHINE
# ==========================================================
train_coming = False
train_seen = False

gate_status = "OPEN"
last_gate_status = ""

vehicle_in_roi = False
train_in_roi = False

train_lost_counter = 0

# ==========================================================
# FUNGSI HITUNG OVERLAP
# ==========================================================
def calculate_overlap(x1, y1, x2, y2):

    bbox_polygon = box(x1, y1, x2, y2)

    intersection_area = roi_polygon.intersection(bbox_polygon).area

    bbox_area = bbox_polygon.area

    if bbox_area == 0:
        return 0

    overlap = 100 * intersection_area / bbox_area

    return overlap

def calculate_train_overlap(x1,y1,x2,y2):

    bbox_polygon = box(x1,y1,x2,y2)

    intersection_area = roi_train_polygon.intersection(
        bbox_polygon
    ).area

    bbox_area = bbox_polygon.area

    if bbox_area == 0:
        return 0

    return 100 * intersection_area / bbox_area

# ==========================================================
# FUNGSI KIRIM SERIAL KE STM32
# ==========================================================
def send_gate_command(cmd):

    global last_gate_status

    # kirim hanya jika status berubah
    if cmd == last_gate_status:
        return

    if ser is not None and ser.is_open:
        try:
            ser.write((cmd + "\n").encode())
            print("TX :", cmd)
        except Exception as e:
            print("Serial send error :", e)

    last_gate_status = cmd


# ==========================================================
# FUNGSI UPDATE GATE STATE
# ==========================================================
def update_gate_state(train_coming, vehicle_in_roi):

    if not train_coming:

        gate_status = "OPEN"

    else:

        if vehicle_in_roi:

            gate_status = "BRAKE"

        else:

            gate_status = "CLOSED"

    return gate_status


# ==========================================================
# WARNA STATUS
# ==========================================================
def gate_color(status):

    if status == "OPEN":
        return (0,255,0)

    elif status == "BRAKE":
        return (0,255,255)

    else:
        return (0,0,255)


# ==========================================================
# STATUS KERETA
# ==========================================================
def train_text(train_coming):

    if train_coming:
        return "TRAIN COMING"

    return "NO TRAIN"

# ==========================================================
# FPS COUNTER
# ==========================================================
fps = 0
fps_counter = 0
fps_timer = time.perf_counter()

cv2.namedWindow("DETEKSI PERLINTASAN", cv2.WINDOW_NORMAL)

# ==========================================================
# LOOP UTAMA
# ==========================================================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    # ===============================
    # HITUNG FPS
    # ===============================
    fps_counter += 1

    now = time.perf_counter()

    if now - fps_timer >= 1.0:
        fps = fps_counter / (now - fps_timer)
        fps_counter = 0
        fps_timer = now

    display = frame.copy()

    # reset status setiap frame
    vehicle_in_roi = False
    train_in_roi = False

    # ======================================================
    # DETEKSI YOLO
    # ======================================================
    if torch.cuda.is_available():
        results = model(
            frame,
            device=0,
            imgsz=640,
            half=True,
            verbose=False
        )
    else:
        results = model(
            frame,
            imgsz=640,
            verbose=False
        )

    for result in results:

        for box_data in result.boxes:

            cls_id = int(box_data.cls[0])
            conf = float(box_data.conf[0])

            # Abaikan confidence rendah
            if conf < CONF_THRESH:
                continue

            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box_data.xyxy[0])

            if label == train_label:

                overlap_train = calculate_train_overlap(
                    x1, y1, x2, y2
                )

                if overlap_train > 0:
                    train_in_roi = True
                    train_seen = True

                cv2.rectangle(
                    display,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 255),
                    2
                )

                text = f"{label} {conf:.2f} {overlap_train:.1f}%"

                cv2.putText(
                    display,
                    text,
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

                continue

            if label not in vehicle_labels:
                continue

            overlap = calculate_overlap(x1, y1, x2, y2)

            # kendaraan dianggap berada di ROI
            if overlap >= OVERLAP_THRESHOLD:
                vehicle_in_roi = True

            # warna bbox
            color = (255,0,0)

            cv2.rectangle(
                display,
                (x1,y1),
                (x2,y2),
                color,
                2
            )

            text = f"{label} {conf:.2f} {overlap:.1f}%"

            cv2.putText(
                display,
                text,
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # ======================================================
    # MONITOR KERETA
    # ======================================================
    if train_seen:

        if train_in_roi:
            train_lost_counter = 0

        else:
            train_lost_counter += 1

        if train_lost_counter >= TRAIN_LOST_FRAME:

            train_coming = False
            train_seen = False
            train_lost_counter = 0

            print("TRAIN CLEARED")

    # ======================================================
    # UPDATE GATE STATUS
    # ======================================================
    gate_status = update_gate_state(
        train_coming,
        vehicle_in_roi
    )

    # ======================================================
    # KIRIM KE STM32
    # ======================================================
    send_gate_command(gate_status)

    # ======================================================
    # ROI
    # ======================================================
    cv2.polylines(
        display,
        [roi_np],
        True,
        (0,255,0),
        2
    )

    cv2.polylines(
        display,
        [roi_train_np],
        True,
        (0,0,255),
        2
    )

    # ======================================================
    # TRAIN STATUS
    # ======================================================
    cv2.putText(
        display,
        f"TRAIN STATUS : {train_text(train_coming)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0,255,255),
        2
    )

    # ======================================================
    # GATE STATUS
    # ======================================================
    cv2.putText(
        display,
        f"GATE STATUS : {gate_status}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        gate_color(gate_status),
        2
    )

    cv2.putText(
        display,
        f"FPS : {fps:.2f}",
        (20,120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    # ======================================================
    # TAMPILKAN
    # ======================================================
    cv2.imshow("DETEKSI PERLINTASAN", display)

    # simpan video
    out.write(display)

    # ======================================================
    # KEYBOARD
    # ======================================================
    key = cv2.waitKey(1) & 0xFF

    # K = kereta datang
    if key == ord('k'):

        train_coming = True
        train_seen = False
        train_lost_counter = 0

        print("TRAIN COMING")

    # Q = keluar
    elif key == ord('q'):
        break


# ==========================================================
# RELEASE
# ==========================================================
cap.release()

out.release()

if ser is not None and ser.is_open:
    ser.close()

cv2.destroyAllWindows()