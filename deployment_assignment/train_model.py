"""
Car Insurance Risk Classifier — Training Script
================================================
Dataset : data.csv  (120 synthetic rows)
Target  : insurance_risk  →  Low | Medium | High
Model   : RandomForestClassifier
Output  : model.pkl  (model + label encoder bundled)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

FEATURE_COLS = [
    "age",
    "driving_experience",
    "accidents_last_5_years",
    "vehicle_age",
    "annual_km",
]
TARGET_COL = "insurance_risk"


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return features + encoded target."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows from {path.name}")
    print(f"[INFO] Target distribution:\n{df[TARGET_COL].value_counts()}\n")

    X = df[FEATURE_COLS]
    y_raw = df[TARGET_COL]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y, le


def train(X_train: pd.DataFrame, y_train: np.ndarray) -> RandomForestClassifier:
    """Train Random Forest with fixed seed."""
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    print("[INFO] Training complete.")
    return clf


def evaluate(clf, le, X_test, y_test) -> None:
    """Print classification metrics."""
    y_pred = clf.predict(X_test)
    print("\n── Classification Report ──────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("── Confusion Matrix ───────────────────────────────────────")
    print(confusion_matrix(y_test, y_pred))
    print()


def save_model(clf, le, path: Path) -> None:
    """Save model + label encoder as a single pickle bundle."""
    bundle = {"model": clf, "label_encoder": le, "features": FEATURE_COLS}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"[INFO] Model saved → {path}")


def main() -> None:
    X, y, le = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    clf = train(X_train, y_train)
    evaluate(clf, le, X_test, y_test)
    save_model(clf, le, MODEL_PATH)


if __name__ == "__main__":
    main()
