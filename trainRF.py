"""
trainRF.py
-----------
Model 2: Random Forest classifier (replaces trainTorch.py from the Lab7
template), as the second architecture for comparison against SVM in part 3
of the report.

Loads ./dataSet/{rock,scissors,paper,error}.csv, trains a RandomForestClassifier
on the 63-d normalized hand-landmark features, and reports:
    - Accuracy
    - Precision / Recall / F1-score (per-class + macro average)
    - Confusion matrix
    - Feature importances (top 10) - useful for the report's discussion section

Saves the trained model to rf_model.pkl
"""

import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from dataset_utils import CLASSES, load_dataset


def main():
    X, y = load_dataset()
    print(f"Loaded {len(X)} samples, {X.shape[1]} features.")
    for c in CLASSES:
        print(f"  {c}: {np.sum(y == c)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=None, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n=== Random Forest results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall    (macro): {rec_macro:.4f}")
    print(f"F1-score  (macro): {f1_macro:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    labels_present = [c for c in CLASSES if c in set(y)]
    print("Confusion matrix (rows=true, cols=pred), order =", labels_present)
    print(confusion_matrix(y_test, y_pred, labels=labels_present))

    print("\nTop 10 most important features (landmark_idx*3 + axis):")
    importances = model.feature_importances_
    top10 = np.argsort(importances)[::-1][:10]
    for idx in top10:
        landmark = idx // 3
        axis = ["x", "y", "z"][idx % 3]
        print(f"  landmark {landmark:2d} {axis}: {importances[idx]:.4f}")

    with open("rf_model.pkl", "wb") as f:
        pickle.dump({"model": model}, f)
    print("\nSaved model -> rf_model.pkl")


if __name__ == "__main__":
    main()
