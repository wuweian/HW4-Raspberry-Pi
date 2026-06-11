"""
cameraStartRPS.py
-------------------
Part 2 deliverable: connects the Raspberry Pi camera feed to the trained
gesture-classification model (SVM or Random Forest), classifies each frame
as Rock(石頭) / Scissors(剪刀) / Paper(布) / Error(其他錯誤手勢), and posts
the annotated frame + result to the Flask app (app.py) for display.

This follows the same pattern as cameraStartSVM.py / cameraStartTorch.py in
the Lab7 (OldManFalls) template, but:
  - uses MediaPipe HANDS instead of Holistic
  - loads svm_model.pkl or rf_model.pkl (whichever scored better, see report)
  - adds an "Error" class for unrecognized gestures / low-confidence predictions

Usage:
    $ python cameraStartRPS.py --model svm     # use svm_model.pkl
    $ python cameraStartRPS.py --model rf      # use rf_model.pkl
    $ python cameraStartRPS.py --model svm --video path/to/video.mp4   # 附錄: 用影片作為輸入

Before running:
  1. Make sure app.py is running (python app.py) so /post_camera_frame exists.
  2. Edit POST_URL below to point at your Flask server's IP (default:
     localhost, change to the Pi's external IP if running on a different
     machine, per the Lab7 instructions: http://<your-ip>:5000/post_camera_frame)
"""

import argparse
import base64
import pickle
import time

import cv2
import mediapipe as mp
import numpy as np
import requests

from MediapipeDataCollect import normalize_landmarks

# --------------------------------------------------------------------------
# 修改這裡的 IP 位址為你的對外 ip (per Lab7 step 4-2)
POST_URL = "http://127.0.0.1:5000/post_camera_frame"
# --------------------------------------------------------------------------

LABEL_DISPLAY = {
    "rock": "Rock",
    "scissors": "Scissors",
    "paper": "Paper",
    "error": "Error",
}

# Confidence threshold below which we report "Error" regardless of the
# model's predicted class (handles gestures the model has never seen).
CONF_THRESHOLD = 0.5

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


def load_model(model_name):
    path = f"{model_name}_model.pkl"
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def predict(bundle, model_name, features):
    model = bundle["model"]
    X = features.reshape(1, -1)

    if model_name == "svm":
        X = bundle["scaler"].transform(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        best_idx = int(np.argmax(proba))
        return classes[best_idx], float(proba[best_idx])
    else:
        pred = model.predict(X)[0]
        return pred, 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["svm", "rf"], default="svm",
        help="which trained model to use (svm_model.pkl or rf_model.pkl)"
    )
    parser.add_argument(
        "--video", default=None,
        help="optional path to a video file instead of the live camera "
             "(附錄: 改為用影片作為輸入)"
    )
    args = parser.parse_args()

    bundle = load_model(args.model)
    print(f"Loaded {args.model}_model.pkl")

    source = args.video if args.video else 0
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Cannot open camera/video")
        return

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while cap.isOpened():
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break

            img = cv2.resize(img, (640, 480))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            label = "error"

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                features = normalize_landmarks(hand_landmarks)
                pred_label, confidence = predict(bundle, args.model, features)

                label = pred_label if confidence >= CONF_THRESHOLD else "error"

            display_text = LABEL_DISPLAY.get(label, "Error")

            cv2.putText(
                img,
                f"Result: {display_text}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.imshow("RPS Detector", img)

            # ---- post frame + result to Flask app ----
            _, buffer = cv2.imencode(".jpg", img)
            jpg_b64 = base64.b64encode(buffer).decode("utf-8")
            payload = {
                "image": jpg_b64,
                "result": display_text,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                requests.post(POST_URL, json=payload, timeout=0.5)
            except requests.exceptions.RequestException as e:
                print(f"[warn] could not reach Flask server: {e}")

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
