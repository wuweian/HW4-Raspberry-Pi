"""
trainSVM.py
------------
Model 1: Support Vector Machine (SVM), following the Lab7 trainSVC.py pattern.

Loads ./dataSet/{rock,scissors,paper,error}.csv, trains an SVM classifier on
the 63-d normalized hand-landmark features, and reports:
    - Accuracy
    - Precision / Recall / F1-score (per-class + macro average)
    - Confusion matrix

Saves the trained model + label encoder to svm_model.pkl
"""

import pickle

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset_utils import CLASSES, load_dataset


def main():
    X, y = load_dataset()
    print(f"Loaded {len(X)} samples, {X.shape[1]} features.")
    for c in CLASSES:
        print(f"  {c}: {np.sum(y == c)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # RBF kernel SVM (typical default for landmark-based gesture classification)
    model = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n=== SVM (RBF kernel) results ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision (macro): {prec_macro:.4f}")
    print(f"Recall    (macro): {rec_macro:.4f}")
    print(f"F1-score  (macro): {f1_macro:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    print("Confusion matrix (rows=true, cols=pred), order =", CLASSES)
    labels_present = [c for c in CLASSES if c in set(y)]
    print(confusion_matrix(y_test, y_pred, labels=labels_present))

    with open("svm_model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)
    print("\nSaved model -> svm_model.pkl")


if __name__ == "__main__":
    main()
