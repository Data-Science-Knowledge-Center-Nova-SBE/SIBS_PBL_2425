import cv2
import time
from ultralytics import YOLO
from dotenv import load_dotenv
import os

model = YOLO("best.pt")

print("Model classes:", model.names)

load_dotenv()
url = os.getenv("url")
print(url)

cap = cv2.VideoCapture(url)
cv2.namedWindow("Carcavelos Tracker", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Carcavelos Tracker", 960, 540)

target_label = "Surfer"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream interrupted — retrying...")
        time.sleep(2)
        cap.release()
        cap = cv2.VideoCapture(url)
        continue

    results = model(frame, conf=0.1, verbose=False)
    
    current_surfers = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            #cv2.putText()
            
            if label == target_label:
                current_surfers += 1
                
    counter_text = f"Surfers detected: {current_surfers}"
    cv2.putText(frame, counter_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Carcavelos Tracker", frame)

    if cv2.waitKey(30) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
