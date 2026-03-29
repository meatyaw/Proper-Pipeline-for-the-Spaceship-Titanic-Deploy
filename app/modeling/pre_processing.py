import pandas as pd
from pathlib import Path

SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Cabin → Deck, Cabin_num, Side
    df["Deck"]      = df["Cabin"].apply(lambda x: x.split("/")[0] if pd.notna(x) else "Unknown")
    df["Cabin_num"] = df["Cabin"].apply(lambda x: float(x.split("/")[1]) if pd.notna(x) else -1.0)
    df["Side"]      = df["Cabin"].apply(lambda x: x.split("/")[2] if pd.notna(x) else "Unknown")

    # Group
    df["Group"]      = df["PassengerId"].apply(lambda x: x.split("_")[0])
    df["Group_size"] = df.groupby("Group")["Group"].transform("count")
    df["Solo"]       = (df["Group_size"] == 1).astype(int)

    # Spending
    df["TotalSpending"] = df[SPENDING_COLS].sum(axis=1)
    df["NoSpending"]    = (df["TotalSpending"] == 0).astype(int)

    return df


def load_featured(path: str | Path = "data/raw/train.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return engineer_features(df)


if __name__ == "__main__":
    df = load_featured()
    print(df.shape, list(df.columns))
