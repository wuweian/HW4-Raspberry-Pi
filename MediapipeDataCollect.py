"""
MediapipeDataCollect.py
------------------------
Adapted from the Lab7 (OldManFalls) MediapipeDataCollect.py template, but uses
MediaPipe HANDS instead of Holistic, since RPS (Rock-Paper-Scissors) only needs
one-hand landmarks.

Usage:
    $ python MediapipeDataCollect.py <label>

    <label> must be one of: rock, scissors, paper, error
    (error = any gesture that is NOT rock/scissors/paper, used to train an
     "Error / 其他錯誤手勢" class)

Each run captures landmark snapshots every CAPTURE_INTERVAL seconds and
appends one row per snapshot to ./dataSet/<label>.csv

Each row = 21 hand landmarks * 3 coords (x, y, z) = 63 features,
normalized relative to the wrist (landmark 0) so the model is
robust to where the hand is in the frame.
"""

import csv
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

CLASSES = ["rock", "scissors", "paper", "error"]
CAPTURE_INTERVAL = 0.5  # seconds between samples


def normalize_landmarks(hand_landmarks):
    """Flatten 21 landmarks to a 63-d vector, normalized relative to the wrist
    (landmark 0) and scaled by the hand size, so the features are
    translation/scale invariant."""
    pts = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float64
    )
    wrist = pts[0].copy()
    pts -= wrist  # translate so wrist = origin

    # scale by the distance wrist -> middle finger MCP (landmark 9) so the
    # feature vector is roughly invariant to hand distance from the camera
    scale = np.linalg.norm(pts[9])
    if scale > 1e-6:
        pts /= scale

    return pts.flatten()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CLASSES:
        print(f"Usage: python {os.path.basename(__file__)} <label>")
        print(f"  <label> must be one of: {', '.join(CLASSES)}")
        sys.exit(1)

    label = sys.argv[1]

    os.makedirs("./dataSet", exist_ok=True)
    csv_filename = f"./dataSet/{label}.csv"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        sys.exit()

    time.sleep(2)  # give the camera time to warm up

    print(f"Collecting samples for class '{label}'.")
    print("Hold the gesture steady in front of the camera. Press 'q' to stop.")

    n_saved = 0
    last_capture = time.time()

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

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                now = time.time()
                if now - last_capture >= CAPTURE_INTERVAL:
                    feature_row = normalize_landmarks(hand_landmarks)
                    with open(csv_filename, "a", newline="") as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(feature_row.tolist())
                    n_saved += 1
                    last_capture = now
                    # beep (Windows) to signal a sample was captured
                    try:
                        import winsound

                        winsound.Beep(1000, 150)
                    except ImportError:
                        pass
                    print(f"[{label}] saved sample #{n_saved}")

            cv2.putText(
                img,
                f"label={label}  saved={n_saved}  (q to quit)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("MediapipeDataCollect", img)

            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {n_saved} samples written to {csv_filename}")


if __name__ == "__main__":
    main()
