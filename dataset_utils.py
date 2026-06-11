"""
dataset_utils.py
-----------------
Shared helper for trainSVM.py / trainRF.py.

Expects ./dataSet/ to contain one CSV per class:
    rock.csv, scissors.csv, paper.csv, error.csv
Each row = 63 features (21 landmarks x x,y,z), produced by
MediapipeDataCollect.py.
"""

import os

import numpy as np
import pandas as pd

CLASSES = ["rock", "scissors", "paper", "error"]
LABEL_NAMES_ZH = {
    "rock": "石頭 (Rock)",
    "scissors": "剪刀 (Scissors)",
    "paper": "布 (Paper)",
    "error": "其他錯誤手勢 (Error)",
}


def load_dataset(data_dir="./dataSet"):
    """Load all class CSVs into X (features) and y (string labels)."""
    X_list = []
    y_list = []

    for label in CLASSES:
        path = os.path.join(data_dir, f"{label}.csv")
        if not os.path.exists(path):
            print(f"[warn] {path} not found, skipping class '{label}'")
            continue
        df = pd.read_csv(path, header=None)
        X_list.append(df.values)
        y_list.extend([label] * len(df))

    if not X_list:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Run MediapipeDataCollect.py first."
        )

    X = np.vstack(X_list)
    y = np.array(y_list)
    return X, y
