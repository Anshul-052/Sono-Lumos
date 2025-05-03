import cv2
import numpy as np
import torch
import pyttsx3
import time
import threading
import pytesseract
from ultralytics import YOLO


engine = pyttsx3.init()
engine.setProperty('rate', 150)  

device = 'cpu'
model = YOLO('yolov8n.pt').to(device)


cap = cv2.VideoCapture(0)

prev_time = 0
speech_interval = 2 


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def speak(text):
    """Runs speech feedback in a separate thread to prevent blocking."""
    def run():
        engine.say(text)
        engine.runAndWait()
    
    speech_thread = threading.Thread(target=run)
    speech_thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, stream=True)
    detected_objects = set()
    detected_text = ""

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if conf > 0.5:  
                x1, y1, x2, y2 = map(int, box)
                label = model.names[class_id]
                detected_objects.add(label)

           
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                roi = frame[y1:y2, x1:x2]
                text = pytesseract.image_to_string(roi, lang="eng").strip()
                if text:
                    detected_text += text + " "


    current_time = time.time()
    if current_time - prev_time > speech_interval and detected_objects:
        if detected_text:
            speak(f"I see {', '.join(detected_objects)}. Also, the text reads: {detected_text}")
        else:
            speak(f"I see {', '.join(detected_objects)}")
        prev_time = current_time


    cv2.imshow('Smart Cane Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
