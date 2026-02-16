from ultralytics import YOLO
import cv2
import numpy as np

# -----------------------------
# MODEL
# -----------------------------
model = YOLO("yolov8n.pt")  # veya yolov8n.pt test için

# -----------------------------
# HSV RENK ARALIKLARI
# -----------------------------
# HSV format: H:0-180  S:0-255  V:0-255

color_ranges = {
    "MAVI": {
        "lower": np.array([90, 80, 50]),
        "upper": np.array([130, 255, 255]),
        "box_color": (255, 0, 0)
    },
    "YESIL": {
        "lower": np.array([35, 80, 50]),
        "upper": np.array([85, 255, 255]),
        "box_color": (0, 255, 0)
    },
    "KIRMIZI_1": {  # kırmızı iki aralık
        "lower": np.array([0, 80, 50]),
        "upper": np.array([10, 255, 255]),
        "box_color": (0, 0, 255)
    },
    "KIRMIZI_2": {
        "lower": np.array([170, 80, 50]),
        "upper": np.array([180, 255, 255]),
        "box_color": (0, 0, 255)
    }
}

# -----------------------------
# KAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            roi = frame[y1:y2, x1:x2]
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            detected_color = "BELIRSIZ"
            best_match_ratio = 0
            final_box_color = (255,255,255)

            for color_name, values in color_ranges.items():

                mask = cv2.inRange(
                    hsv_roi,
                    values["lower"],
                    values["upper"]
                )

                # Gürültü temizleme
                kernel = np.ones((5,5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # ROI içindeki renk oranı
                match_ratio = np.sum(mask > 0) / (roi.shape[0] * roi.shape[1])

                # %20’den fazla ise renk kabul
                if match_ratio > 0.20 and match_ratio > best_match_ratio:
                    best_match_ratio = match_ratio

                    if "KIRMIZI" in color_name:
                        detected_color = "KIRMIZI"
                    else:
                        detected_color = color_name

                    final_box_color = values["box_color"]

            cv2.rectangle(frame, (x1,y1), (x2,y2), final_box_color, 2)
            cv2.putText(frame, detected_color, (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, final_box_color, 2)

    cv2.imshow("Drone Bayrak Tespit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()