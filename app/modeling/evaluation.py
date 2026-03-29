import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from pre_processing import load_featured
from pipeline import CAT_FEATURES, NUM_FEATURES

ARTIFACTS_DIR = Path("artifacts")
FEATURES      = CAT_FEATURES + NUM_FEATURES

def evaluate():
    df = load_featured()
    X  = df[FEATURES]
    y  = df["Transported"].astype(int)

    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = joblib.load(ARTIFACTS_DIR / "model.pkl")
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy":  accuracy_score(y_val, preds),
        "precision": precision_score(y_val, preds, average="macro"),
        "recall":    recall_score(y_val, preds, average="macro"),
        "f1":        f1_score(y_val, preds, average="macro"),
        "roc_auc":   roc_auc_score(y_val, proba),
    }

    for name, val in metrics.items():
        print(f"  {name:<12}: {val:.4f}")

    return metrics

if __name__ == "__main__":
    evaluate()
