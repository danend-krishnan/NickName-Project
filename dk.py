import cv2
import os
import numpy as np
import pickle
import random
import subprocess
import threading

# ------------------ CONFIG ------------------
NAME_COLLECTION_FILE = "nickname.txt"  # File with all possible nicknames (one per line)
NAMES_FILE = "nicknames.txt"           # Already used names
FACES_DIR = "faces"
ENCODINGS_FILE = "face_recognizer.yml"
LABELS_FILE = "labels.pkl"
# --------------------------------------------

os.makedirs(FACES_DIR, exist_ok=True)

# Load available nicknames
if os.path.exists(NAME_COLLECTION_FILE):
    with open(NAME_COLLECTION_FILE, "r") as f:
        all_names = [line.strip() for line in f if line.strip()]
else:
    print(f"[ERROR] {NAME_COLLECTION_FILE} not found.")
    exit()

# Load used names
if os.path.exists(NAMES_FILE):
    with open(NAMES_FILE, "r") as f:
        used_names = set(line.strip() for line in f if line.strip())
else:
    used_names = set()

# Remaining available names
available_names = list(set(all_names) - used_names)

# Init recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
label_map = {}

if os.path.exists(ENCODINGS_FILE) and os.path.getsize(ENCODINGS_FILE) > 0 and os.path.exists(LABELS_FILE):
    try:
        face_recognizer.read(ENCODINGS_FILE)
        with open(LABELS_FILE, "rb") as f:
            label_map = pickle.load(f)
    except Exception as e:
        print(f"[WARNING] Could not load recognizer: {e}")
        label_map = {}

next_label_id = max(label_map.keys(), default=-1) + 1

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

print("[INFO] Starting live nickname generator... Press Q to quit.")

# ---------------- FUNCTIONS ----------------
def get_random_nickname():
    global available_names
    if not available_names:
        print("[WARNING] No more nicknames available!")
        return None
    nickname = random.choice(available_names)
    available_names.remove(nickname)
    used_names.add(nickname)
    with open(NAMES_FILE, "a") as f:
        f.write(nickname + "\n")
    return nickname

def retrain_recognizer():
    global face_recognizer
    images, ids = [], []
    for label_id, name in label_map.items():
        img_path = os.path.join(FACES_DIR, f"{name}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                ids.append(label_id)
    if images:
        face_recognizer.train(images, np.array(ids))
        face_recognizer.save(ENCODINGS_FILE)
        with open(LABELS_FILE, "wb") as f:
            pickle.dump(label_map, f)

def async_retrain():
    threading.Thread(target=retrain_recognizer, daemon=True).start()

# ----------------- MAIN LOOP -----------------
frame_count = 0
last_seen_names = {}  # (x,y,w,h) → (nickname, frames_left)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        break

    # Resize frame for faster detection
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

    frame_count += 1
    if frame_count % 3 == 0:  # Detect every 3rd frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    else:
        faces = []

    current_faces = []

    for (x, y, w, h) in faces:
        # Scale back coords
        x, y, w, h = x*2, y*2, w*2, h*2
        face_img = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        nickname = None

        # Recognizer check
        if len(label_map) > 0:
            try:
                label_id, confidence = face_recognizer.predict(face_img)
                if confidence < 70:
                    nickname = label_map[label_id]
            except:
                pass

        # New face
        if nickname is None:
            nickname = get_random_nickname()
            if nickname:
                cv2.imwrite(os.path.join(FACES_DIR, f"{nickname}.jpg"), face_img)
                label_map[next_label_id] = nickname
                next_label_id += 1
                async_retrain()

        if nickname:
            current_faces.append(((x, y, w, h), nickname))
            last_seen_names[(x, y, w, h)] = [nickname, 10]  # Show for 10 more frames

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Decrease "frames_left" counter for remembered names
    for (box, data) in list(last_seen_names.items()):
        nickname, ttl = data
        if ttl > 0:
            (x, y, w, h) = box
            cv2.putText(frame, nickname, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            last_seen_names[box][1] -= 1
        else:
            del last_seen_names[box]

    cv2.imshow("Nickname Generator", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
