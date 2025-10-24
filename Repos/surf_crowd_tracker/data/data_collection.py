import cv2, time, os
from dotenv import load_dotenv

load_dotenv()
url = os.getenv("url")

cap = cv2.VideoCapture(url)
os.makedirs("frames", exist_ok=True)

i = 0
last_save = time.time()

TOP = 0.0
BOTTOM = 1.0
LEFT = 0.17
RIGHT = 0.83

while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream interrupted — reconnecting...")
        time.sleep(2)
        cap.release()
        cap = cv2.VideoCapture(url)
        continue

    now = time.time()
    if now - last_save >= 5:
        print("Start Taking a picture")
        h, w = frame.shape[:2]

        y1, y2 = int(h * TOP), int(h * BOTTOM)
        x1, x2 = int(w * LEFT), int(w * RIGHT)
        roi = frame[y1:y2, x1:x2]

        zoomed = cv2.resize(roi, (w, h), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(f"frames/frame_{i:04d}.jpg", zoomed)
        print(f"Saved frame {i} at zoom region y:{y1}-{y2}, x:{x1}-{x2}")
        i += 1
        last_save = now 
