from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

CAT_FEATURES = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
NUM_FEATURES = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "Cabin_num", "Group_size", "Solo", "TotalSpending", "NoSpending",
]

cat_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])

num_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

preprocessor = ColumnTransformer([
    ("cat", cat_transformer, CAT_FEATURES),
    ("num", num_transformer, NUM_FEATURES),
])

def get_pipeline() -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   LogisticRegression(max_iter=1000, random_state=42)),
    ])
