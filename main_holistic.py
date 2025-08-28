from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64, numpy as np, cv2, mediapipe as mp
from keras.models import load_model
from collections import deque
from typing import List, Optional, Dict
from label import LABELS
import math

MIN_FRAME_FOR_PREDICTION = 30
BUFFER_SIZE = 150  # time steps untuk TCN input

app = FastAPI()

model = load_model("models/holistic_tcn_model.h5")
dpose_buffer = deque(maxlen=BUFFER_SIZE)

mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose_detector = mp_holistic.Holistic(static_image_mode=False)

# ---------- Utils geometry ----------
def _angle(a, b, c):
    # angle ABC (derajat)
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cosang = np.clip(cosang, -1.0, 1.0)
    return math.degrees(math.acos(cosang))

def _dist(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.linalg.norm(a - b))

def _pose_xyz(flat_pts, name):
    idx = mp_pose.PoseLandmark[name].value
    base = idx * 3
    return [flat_pts[base], flat_pts[base+1], flat_pts[base+2]]

def _hand_xyz(flat_pts, is_left, name):
    # urutan: 33 pose, 21 left hand, 21 right hand
    base = 33*3 + (0 if is_left else 21*3)
    idx = mp_hands.HandLandmark[name].value
    b = base + idx*3
    return [flat_pts[b], flat_pts[b+1], flat_pts[b+2]]

def _to_2d_list(flat_pts, count):
    # ambil x,y untuk tiap landmark
    out = []
    for i in range(count):
        out.extend([float(flat_pts[i*3]), float(flat_pts[i*3+1])])
    return out

def _shoulder_width(flat_pts):
    ls = _pose_xyz(flat_pts, "LEFT_SHOULDER")
    rs = _pose_xyz(flat_pts, "RIGHT_SHOULDER")
    return max(_dist(ls, rs), 1e-6)  # untuk skala jarak

def _finger_curl_score(flat_pts, is_left):
    """Skor 0..1: 1 artinya mengepal/tertekuk (fist), 0 artinya lurus membuka.
       Naif: rata-rata sudut di MCP antara tulang (MCP-PIP-DIP) tiap jari."""
    mcp = mp_hands.HandLandmark.MCP
    pip = mp_hands.HandLandmark.PIP
    dip = mp_hands.HandLandmark.DIP
    finger_ids = [
        (mp_hands.HandLandmark.INDEX_MCP, mp_hands.HandLandmark.INDEX_PIP, mp_hands.HandLandmark.INDEX_DIP),
        (mp_hands.HandLandmark.MIDDLE_MCP, mp_hands.HandLandmark.MIDDLE_PIP, mp_hands.HandLandmark.MIDDLE_DIP),
        (mp_hands.HandLandmark.RING_MCP, mp_hands.HandLandmark.RING_PIP, mp_hands.HandLandmark.RING_DIP),
        (mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_DIP),
    ]
    curls = []
    try:
        for mcp_id, pip_id, dip_id in finger_ids:
            a = _hand_xyz(flat_pts, is_left, mcp_id.name)
            b = _hand_xyz(flat_pts, is_left, pip_id.name)
            c = _hand_xyz(flat_pts, is_left, dip_id.name)
            ang = _angle(a, b, c)  # kecil -> tertekuk
            curls.append(max(0.0, min(1.0, (180 - ang)/180)))  # 0..1
    except:
        return None
    if not curls:
        return None
    return float(np.mean(curls))

# ---------- Feedback schema ----------
class JointFeedback(BaseModel):
    part: str
    message: str
    score: float  # 0..1 (semakin besar = semakin “salah”)
    landmark_indices: List[int] = []  # index pose/hand utk highlight

class PredictResponse(BaseModel):
    label: str
    confidence: float
    feedback: List[JointFeedback] = []
    pose_2d: Optional[List[float]] = None
    left_hand_2d: Optional[List[float]] = None
    right_hand_2d: Optional[List[float]] = None

# ---------- Aturan per gerakan (contoh) ----------
# target ± toleransi; score ~ deviasi/ toleransi, dipotong 0..1
def _score_from_target(measured, target, tol):
    dev = abs(measured - target)
    return float(max(0.0, min(1.0, dev / tol)))

# Contoh label, sesuaikan dg isi LABELS-mu
LBL_PUNCH = "oi_zuki"      # straight punch
LBL_FRONT_KICK = "mae_geri"

def _feedback_oi_zuki(flat_pts) -> List[JointFeedback]:
    fb = []
    S = _shoulder_width(flat_pts)
    # siku kanan/lurus (anggap pukulan kanan)
    re = _pose_xyz(flat_pts, "RIGHT_ELBOW")
    rs = _pose_xyz(flat_pts, "RIGHT_SHOULDER")
    rw = _pose_xyz(flat_pts, "RIGHT_WRIST")
    ang_re = _angle(rs, re, rw)  # 180 ~ lurus
    sc = _score_from_target(ang_re, 175, 15)
    if sc > 0.3:
        fb.append(JointFeedback(
            part="Siku kanan",
            message="Luruskan siku pukulan (extend lebih).",
            score=sc,
            landmark_indices=[mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                              mp_pose.PoseLandmark.RIGHT_ELBOW.value,
                              mp_pose.PoseLandmark.RIGHT_WRIST.value]
        ))
    # tangan kiri di “chamber” (wrist kiri dekat pinggang kiri)
    lw = _pose_xyz(flat_pts, "LEFT_WRIST")
    lh = _pose_xyz(flat_pts, "LEFT_HIP")
    d = _dist(lw, lh) / S
    sc = max(0.0, min(1.0, (d - 0.25) / 0.25))  # >0.25*shoulder dianggap melenceng
    if sc > 0.3:
        fb.append(JointFeedback(
            part="Tangan kiri",
            message="Simpan tangan non-pukulan di pinggang (chamber).",
            score=sc,
            landmark_indices=[mp_pose.PoseLandmark.LEFT_WRIST.value,
                              mp_pose.PoseLandmark.LEFT_HIP.value]
        ))
    # kepalan: curl jari kanan & kiri
    curl_r = _finger_curl_score(flat_pts, is_left=False)
    if curl_r is not None:
        sc = max(0.0, min(1.0, (0.8 - curl_r)/0.4))  # target ~0.8 (mengepalkan)
        if sc > 0.3:
            fb.append(JointFeedback(
                part="Kepalan kanan",
                message="Kepalkan tangan lebih rapat saat pukulan.",
                score=sc,
                landmark_indices=[]  # pakai indeks hand kanan di sisi Flutter
            ))
    curl_l = _finger_curl_score(flat_pts, is_left=True)
    if curl_l is not None:
        sc = max(0.0, min(1.0, (0.8 - curl_l)/0.4))
        if sc > 0.3:
            fb.append(JointFeedback(
                part="Kepalan kiri",
                message="Kepalkan tangan kiri rapat di chamber.",
                score=sc
            ))
    # bahu sejajar (optional)
    ls = _pose_xyz(flat_pts, "LEFT_SHOULDER")
    rs = _pose_xyz(flat_pts, "RIGHT_SHOULDER")
    shoulder_level = abs(ls[1] - rs[1])
    sc = max(0.0, min(1.0, (shoulder_level - 0.03) / 0.05))
    if sc > 0.3:
        fb.append(JointFeedback(part="Bahu", message="Jaga bahu tetap sejajar.", score=sc,
                                landmark_indices=[mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                                  mp_pose.PoseLandmark.RIGHT_SHOULDER.value]))
    return fb

def _feedback_mae_geri(flat_pts) -> List[JointFeedback]:
    fb = []
    S = _shoulder_width(flat_pts)
    # anggap kaki kanan menendang
    rh = _pose_xyz(flat_pts, "RIGHT_HIP")
    rk = _pose_xyz(flat_pts, "RIGHT_KNEE")
    ra = _pose_xyz(flat_pts, "RIGHT_ANKLE")
    ang_knee = _angle(rh, rk, ra)  # ~180 saat impact
    sc = _score_from_target(ang_knee, 170, 20)
    if sc > 0.3:
        fb.append(JointFeedback(part="Lutut kanan",
                                message="Luruskan lutut kaki penendang saat impact.",
                                score=sc,
                                landmark_indices=[mp_pose.PoseLandmark.RIGHT_HIP.value,
                                                  mp_pose.PoseLandmark.RIGHT_KNEE.value,
                                                  mp_pose.PoseLandmark.RIGHT_ANKLE.value]))
    # tinggi tendangan: ankle > tinggi pinggang
    height_ok = (ra[1] < rh[1] - 0.02)  # y lebih kecil = lebih tinggi
    if not height_ok:
        fb.append(JointFeedback(part="Ketinggian tendangan",
                                message="Angkat kaki lebih tinggi (minimal setara pinggang).",
                                score=0.6,
                                landmark_indices=[mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                                                  mp_pose.PoseLandmark.RIGHT_HIP.value]))
    # tumit lurus (arah kaki)
    rheel = _pose_xyz(flat_pts, "RIGHT_HEEL")
    # arah tumit ke depan: jarak x/z dari heel ke toe lebih kecil dari hip? (naif)
    # Cek saja ankle–heel hampir sejajar garis kaki (optional sederhana)
    d_ah = _dist(ra, rheel)/S
    sc = max(0.0, min(1.0, (0.25 - d_ah)/0.25))  # ingin heel dekat ankle
    if sc > 0.3:
        fb.append(JointFeedback(part="Tumit kaki",
                                message="Kunci tumit sejajar dengan pergelangan saat menendang.",
                                score=sc,
                                landmark_indices=[mp_pose.PoseLandmark.RIGHT_ANKLE.value,
                                                  mp_pose.PoseLandmark.RIGHT_HEEL.value]))
    return fb

def _compute_feedback(label, flat_pts) -> List[JointFeedback]:
    if label == LBL_PUNCH:
        return _feedback_oi_zuki(flat_pts)
    if label == LBL_FRONT_KICK:
        return _feedback_mae_geri(flat_pts)
    # fallback: cek umum
    fb = []
    # contoh umum: punggung tegak (pinggul-bahu vertikal)
    lh = _pose_xyz(flat_pts, "LEFT_HIP"); rh = _pose_xyz(flat_pts, "RIGHT_HIP")
    ls = _pose_xyz(flat_pts, "LEFT_SHOULDER"); rs = _pose_xyz(flat_pts, "RIGHT_SHOULDER")
    mid_hip = [(lh[0]+rh[0])/2, (lh[1]+rh[1])/2, (lh[2]+rh[2])/2]
    mid_sh  = [(ls[0]+rs[0])/2, (ls[1]+rs[1])/2, (ls[2]+rs[2])/2]
    # “ketegakan” diukur dari offset x antara mid-hip dan mid-shoulder
    sc = max(0.0, min(1.0, (abs(mid_sh[0]-mid_hip[0]) - 0.02)/0.05))
    if sc > 0.3:
        fb.append(JointFeedback(part="Punggung",
                                message="Jaga punggung lebih tegak.",
                                score=sc,
                                landmark_indices=[mp_pose.PoseLandmark.LEFT_HIP.value,
                                                  mp_pose.PoseLandmark.RIGHT_HIP.value,
                                                  mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                                                  mp_pose.PoseLandmark.RIGHT_SHOULDER.value]))
    return fb

# ---------- Endpoint ----------
class ImageRequest(BaseModel):
    image: str

@app.post("/predict", response_model=PredictResponse)
async def predict_frame(req: ImageRequest):
    try:
        image_bytes = base64.b64decode(req.image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = pose_detector.process(img_rgb)

        pts = []
        pose_2d = []
        lhand_2d = []
        rhand_2d = []

        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                pts.extend([lm.x, lm.y, lm.z])
                pose_2d.extend([lm.x, lm.y])
        else:
            pts.extend([0]*(33*3))
            pose_2d = [0.0]*(33*2)

        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                pts.extend([lm.x, lm.y, lm.z])
                lhand_2d.extend([lm.x, lm.y])
        else:
            pts.extend([0]*(21*3))
            lhand_2d = [0.0]*(21*2)

        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                pts.extend([lm.x, lm.y, lm.z])
                rhand_2d.extend([lm.x, lm.y])
        else:
            pts.extend([0]*(21*3))
            rhand_2d = [0.0]*(21*2)

        # kalau semua 0, jangan prediksi
        if sum(pts) == 0:
            return PredictResponse(label="-", confidence=0.0)

        dpose_buffer.append(pts)

        if len(dpose_buffer) < MIN_FRAME_FOR_PREDICTION:
            return PredictResponse(label="-", confidence=0.0)

        buf_np = np.array(dpose_buffer)
        T = buf_np.shape[0]
        if T < BUFFER_SIZE:
            pad = np.zeros((BUFFER_SIZE - T, buf_np.shape[1]))
            input_seq = np.vstack([pad, buf_np])
        else:
            input_seq = buf_np[-BUFFER_SIZE:]

        input_seq = input_seq.reshape(1, BUFFER_SIZE, buf_np.shape[1])
        probs = model.predict(input_seq, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = LABELS[idx]
        confidence = float(probs[idx])

        # hitung feedback pada frame terakhir yang terkini
        flat_pts = dpose_buffer[-1]
        feedback = _compute_feedback(label, flat_pts)

        return PredictResponse(
            label=label,
            confidence=confidence,
            feedback=feedback,
            pose_2d=pose_2d,
            left_hand_2d=lhand_2d,
            right_hand_2d=rhand_2d
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))