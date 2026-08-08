import cv2
import json
import numpy as np

VIDEO_SOURCE = "perlintasan.mp4"

ROI_VEHICLE_FILE = "roi_kendaraan.json"
ROI_TRAIN_FILE = "roi_kereta.json"

points = []
mode = "VEHICLE"   # VEHICLE -> TRAIN


def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Titik ditambahkan: ({x}, {y})")


cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("ERROR: Kamera tidak dapat dibuka.")
    exit()

cv2.namedWindow("BUILD ROI", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("BUILD ROI", mouse_callback)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    display = frame.copy()

    # ==========================================
    # Warna dan teks sesuai mode
    # ==========================================
    if mode == "VEHICLE":
        color = (0, 255, 0)
        title = "BUILD ROI KENDARAAN"
    else:
        color = (0, 0, 255)
        title = "BUILD ROI KERETA"

    # ==========================================
    # Gambar titik
    # ==========================================
    for p in points:
        cv2.circle(display, p, 5, color, -1)

    # ==========================================
    # Gambar polygon
    # ==========================================
    if len(points) > 1:
        pts = np.array(points, np.int32)
        cv2.polylines(display, [pts], False, color, 2)

    if len(points) >= 3:
        pts = np.array(points, np.int32)
        cv2.polylines(display, [pts], True, color, 2)

    # ==========================================
    # Informasi
    # ==========================================
    cv2.putText(display,
                title,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2)

    cv2.putText(display,
                "ENTER = Simpan | R = Reset | Q = Keluar",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2)

    cv2.imshow("BUILD ROI", display)

    key = cv2.waitKey(1) & 0xFF

    # ==========================================
    # ENTER
    # ==========================================
    if key == 13:

        if len(points) < 3:
            print("ROI minimal 3 titik.")

        else:

            data = {
                "points": points
            }

            # -----------------------------
            # ROI kendaraan
            # -----------------------------
            if mode == "VEHICLE":

                with open(ROI_VEHICLE_FILE, "w") as f:
                    json.dump(data, f, indent=4)

                print("ROI kendaraan disimpan.")

                # pindah ke ROI kereta
                mode = "TRAIN"
                points = []

            # -----------------------------
            # ROI kereta
            # -----------------------------
            else:

                with open(ROI_TRAIN_FILE, "w") as f:
                    json.dump(data, f, indent=4)

                print("ROI kereta disimpan.")
                print("SELESAI")

                break

    # ==========================================
    # RESET
    # ==========================================
    elif key == ord('r'):
        points = []
        print("ROI direset.")

    # ==========================================
    # KELUAR
    # ==========================================
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()