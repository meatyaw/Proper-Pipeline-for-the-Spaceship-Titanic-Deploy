from pathlib import Path
import pandas as pd

RAW_DIR    = Path("data/raw")
INPUT_FILE = RAW_DIR / "train.csv"

REQUIRED_COLUMNS = {
    "PassengerId", "HomePlanet", "CryoSleep", "Cabin",
    "Destination", "Age", "VIP", "RoomService", "FoodCourt",
    "ShoppingMall", "Spa", "VRDeck", "Name", "Transported",
}

def ingest():
    df = pd.read_csv(INPUT_FILE)
    missing = REQUIRED_COLUMNS - set(df.columns)
    assert not missing, f"Kolom tidak ditemukan: {missing}"
    assert not df.empty, "Dataset kosong."
    print(f"Ingested {len(df)} rows from {INPUT_FILE}")

if __name__ == "__main__":
    ingest()
