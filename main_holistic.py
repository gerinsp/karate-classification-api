from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
import mediapipe as mp
from keras.models import load_model
from collections import deque
from label import LABELS

MIN_FRAME_FOR_PREDICTION = 30
BUFFER_SIZE = 150  # time steps untuk TCN input

app = FastAPI()

model = load_model("models/holistic_tcn_model.h5")

dpose_buffer = deque(maxlen=BUFFER_SIZE)

mp_holistic = mp.solutions.holistic
pose_detector = mp_holistic.Holistic(static_image_mode=False)

class ImageRequest(BaseModel):
    image: str

class PredictResponse(BaseModel):
    label: str
    confidence: float

@app.post("/predict", response_model=PredictResponse)
async def predict_frame(req: ImageRequest):
    """
    Streaming endpoint: append frame keypoints, perform sliding-window prediction.
    Returns prediction every call after minimal frames.
    """
    try:
        image_bytes = base64.b64decode(req.image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose_detector.process(img_rgb)

        pts = []

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                pts.extend([lm.x, lm.y, lm.z])
        else:
            pts.extend([0] * (33 * 3))

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                pts.extend([lm.x, lm.y, lm.z])
        else:
            pts.extend([0] * (21 * 3))

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                pts.extend([lm.x, lm.y, lm.z])
        else:
            pts.extend([0] * (21 * 3))

        if sum(pts) == 0:
            return PredictResponse(label="-", confidence=0.0)

        dpose_buffer.append(pts)

        if len(dpose_buffer) < BUFFER_SIZE:
            return PredictResponse(label="-", confidence=0.0)

        buf_np = np.array(dpose_buffer)
        T = buf_np.shape[0]
        if T < BUFFER_SIZE:
            pad = np.zeros((BUFFER_SIZE - T, buf_np.shape[1]))
            input_seq = np.vstack([pad, buf_np])
        else:
            input_seq = buf_np[-BUFFER_SIZE:]

        input_seq = input_seq.reshape(1, BUFFER_SIZE, buf_np.shape[1])
        probs = model.predict(input_seq)[0]
        idx = int(np.argmax(probs))
        label = LABELS[idx]
        confidence = float(probs[idx])

        return PredictResponse(label=label, confidence=confidence)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
