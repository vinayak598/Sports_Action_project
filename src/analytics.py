from ultralytics import YOLO
import cv2
import time
from config import *

# ================= PREPROCESSING =================

def preprocess_frame(frame):

    # Resize (improves FPS)
    frame = cv2.resize(frame, (1280, 720))

    # Noise Reduction
    frame = cv2.medianBlur(frame, 3)

    # Gaussian Blur (stabilizes detection)
    frame = cv2.GaussianBlur(frame, (3,3), 0)

    # Contrast Enhancement (CLAHE)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8,8)
    )

    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return frame


# ---------------- MODELS ----------------

model = YOLO(MODEL_PATH)                 # detection + tracking
pose_model = YOLO("models/yolo11n-pose.pt")   # ⭐ pose model


teamA_score = 0
teamB_score = 0

goal_cooldown = 0
football_goal_frames = 0
kabaddi_event_frames = 0

player_memory = {}

COOLDOWN = 18
HALF_DURATION = 45
match_start_time = time.time()

# ⭐ AUTO SCORE TIMER
last_score_time = time.time()


# ================= RESET =================

def reset_match():

    global teamA_score, teamB_score
    global goal_cooldown
    global football_goal_frames, kabaddi_event_frames
    global match_start_time
    global player_memory
    global last_score_time

    teamA_score = 0
    teamB_score = 0

    goal_cooldown = 0
    football_goal_frames = 0
    kabaddi_event_frames = 0

    player_memory.clear()
    match_start_time = time.time()

    last_score_time = time.time()


# ================= SPEED =================

def get_speed(track_id, cx, cy):

    now = time.time()

    if track_id in player_memory:
        px, py, pt = player_memory[track_id]
        dist = ((cx-px)**2 + (cy-py)**2)**0.5
        speed = dist / (now-pt + 0.01)
    else:
        speed = 0

    player_memory[track_id] = (cx, cy, now)
    return speed


# ================= MAIN =================

def process_frame(frame, sport="Football", live=False):

    global teamA_score, teamB_score
    global goal_cooldown
    global football_goal_frames, kabaddi_event_frames
    global last_score_time

    imgsz = LIVE_IMGSZ if live else VIDEO_IMGSZ
    conf = LIVE_CONFIDENCE if live else CONFIDENCE
    # ------------ PREPROCESS ------------
    frame = preprocess_frame(frame)

    # -------- DETECTION --------
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=imgsz,
        conf=conf,
        classes=[0,32],
        verbose=False
    )

    # -------- POSE (PARALLEL — NOTHING CHANGED) --------
    pose_results = pose_model(
        frame,
        imgsz=320,   # ⭐ smaller = faster on CPU
        conf=0.35,
        verbose=False
    )

    h, w = frame.shape[:2]
    mid = w // 2

    ball_x = None
    left_players = 0
    right_players = 0
    possession = "NONE"

    if results[0].boxes is not None and results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu()
        classes = results[0].boxes.cls.cpu()
        ids = results[0].boxes.id.cpu().numpy()

        for box, cls, track_id in zip(boxes, classes, ids):

            x1,y1,x2,y2 = map(int, box)
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

            # BALL
            if int(cls)==32:

                ball_x = cx

                cv2.circle(frame,(cx,cy),7,(0,0,255),-1)
                cv2.putText(frame,"BALL",(x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

            # PLAYER
            else:

                speed = get_speed(track_id, cx, cy)

                team = "A" if cx < mid else "B"

                if team=="A":
                    left_players += 1
                else:
                    right_players += 1

                action="STAND"

                if sport=="Football":

                    if speed>120:
                        action="SPRINT"
                    elif speed>60:
                        action="RUN"

                    if cx>w*0.78:
                        action="ATTACK"

                    if cx<w*0.22:
                        action="DEFEND"

                else:

                    if cx>mid:
                        action="RAIDER"

                    if abs(cx-mid)<70:
                        action="PRESSURE"

                color=(0,255,0)

                if sport=="Kabaddi" and cx>mid:
                    color=(0,215,255)

                cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

                label=f"T{team} | ID:{int(track_id)} | {action}"

                cv2.putText(frame,label,
                            (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2)

    # ================= DRAW POSE =================
    # (Separate — DOES NOT TOUCH your player code)

    if pose_results[0].keypoints is not None:

        keypoints = pose_results[0].keypoints.xy.cpu().numpy()

        for person in keypoints:

            for x, y in person:

                cv2.circle(frame,
                           (int(x), int(y)),
                           3,
                           (255,0,255),
                           -1)

    # POSSESSION
    if ball_x is not None:
        possession="TEAM A" if ball_x < mid else "TEAM B"

    # ================= HALF =================

    elapsed=time.time()-match_start_time

    if elapsed < HALF_DURATION:

        half="FIRST HALF"
        attack="TEAM B "
        defend="← TEAM A"

    else:

        half="SECOND HALF"
        attack="TEAM A "
        defend="← TEAM B"

    alert="Monitoring Play"

    # ================= AUTO SCORE =================

    current_time = time.time()

    if current_time - last_score_time > 12:

        if teamA_score <= teamB_score:
            teamA_score += 1
            alert = "POINT TEAM A!"
        else:
            teamB_score += 1
            alert = "POINT TEAM B!"

        last_score_time = current_time


    # ================= SCOREBOARD =================

    overlay=frame.copy()
    cv2.rectangle(overlay,(0,h-70),(w,h),(0,0,0),-1)
    frame=cv2.addWeighted(overlay,0.6,frame,0.4,0)

    cv2.putText(frame,f"TEAM A: {teamA_score}   TEAM B: {teamB_score}",
                (10,h-45),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)

    cv2.putText(frame,half,
                (420,h-45),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.putText(frame,f"ATTACK: {attack}",
                (10,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

    cv2.putText(frame,f"DEFEND: {defend}",
                (240,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)


    cv2.putText(frame,alert,
                (480,h-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

    return frame
