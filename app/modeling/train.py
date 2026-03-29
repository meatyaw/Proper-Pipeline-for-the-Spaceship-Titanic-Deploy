import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from pre_processing import load_featured
from pipeline import get_pipeline, CAT_FEATURES, NUM_FEATURES

ARTIFACTS_DIR = Path("artifacts")
FEATURES      = CAT_FEATURES + NUM_FEATURES

def train():
    df = load_featured()

    X = df[FEATURES]
    y = df["Transported"].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = get_pipeline()
    pipeline.fit(X_train, y_train)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, ARTIFACTS_DIR / "model.pkl")

    val_acc = pipeline.score(X_val, y_val)
    print(f"Val Accuracy: {val_acc:.4f}")
    print(f"Model saved → {ARTIFACTS_DIR / 'model.pkl'}")

if __name__ == "__main__":
    train()
