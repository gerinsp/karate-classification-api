import cv2
import base64
import requests
import os

VIDEO_PATH = r"/Users/gerin/Downloads/Data Karate/2nd brown/Sokumen-Zuki/Sokumen-Zuki Front View.mp4"
API_URL = "http://192.168.55.114:8000/predict"
FRAME_INTERVAL = 1  

if not os.path.exists(VIDEO_PATH):
    print("❌ File tidak ditemukan!")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_count = 0
sent_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % FRAME_INTERVAL == 0:
        _, buf = cv2.imencode(".jpg", frame)
        encoded = base64.b64encode(buf).decode("utf-8")

        payload = { "image": encoded }
        r = requests.post(API_URL, json=payload)
        if not r.ok:
            print(f"Error at frame {frame_count}:", r.text)
        else:
            res = r.json()
            label = res.get("label", "-")
            conf  = res.get("confidence", 0.0)
            print(f"[{sent_count+1:03d}] Label: {label:12s}  Confidence: {conf:.3f}")

        sent_count += 1

    frame_count += 1

cap.release()
