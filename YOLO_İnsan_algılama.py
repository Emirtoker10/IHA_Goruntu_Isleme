from ultralytics import YOLO
import cv2

# Hazır model (person dahil)
model = YOLO("yolov8n.pt")  # Nano versiyon drone için ideal

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])

            # COCO class 0 = person
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame,
                            f"INSAN {conf:.2f}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0,255,0),
                            2)

    cv2.imshow("Insan Tespit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()