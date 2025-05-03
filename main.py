from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np
import threading
import pyttsx3
import pytesseract
import geocoder
from detect_objects import detect_objects

app = FastAPI()


engine = pyttsx3.init()

def speak(text):
    def run_speech():
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=run_speech)
    thread.start()

@app.post("/detect")
async def detect():
    frame = get_camera_frame()
    detected_objects = detect_objects(frame)
    

    if detected_objects:
        speak(f"Detected {', '.join(detected_objects)}")
    
    return JSONResponse(content={"objects": detected_objects})

@app.post("/text_detect")
async def text_detect(file: UploadFile = File(...)):
    contents = await file.read()
    np_image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    text = pytesseract.image_to_string(image)
    
    if text.strip():
        speak(f"Detected text: {text}")
    
    return JSONResponse(content={"text": text})

@app.get("/get_location")
def get_location():
    g = geocoder.ip('me')
    if g.latlng:
        latitude, longitude = g.latlng
        speak(f"Your current location is Latitude: {latitude}, Longitude: {longitude}")
        return JSONResponse(content={"latitude": latitude, "longitude": longitude})
    else:
        return JSONResponse(content={"error": "Unable to fetch location"})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

def get_camera_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()
