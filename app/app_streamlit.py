import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "artifacts" / "model.pkl"

st.set_page_config(page_title="Spaceship Titanic")
st.title("Spaceship Titanic — Prediksi Transportasi")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        home_planet   = st.selectbox("Home Planet",  ["Earth", "Europa", "Mars"])
        cryo_sleep    = st.selectbox("CryoSleep",    [False, True])
        destination   = st.selectbox("Destination",  ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
        vip           = st.selectbox("VIP",          [False, True])
        deck          = st.selectbox("Deck",         ["A", "B", "C", "D", "E", "F", "G", "T"])
        side          = st.selectbox("Side",         ["P", "S"])
        age           = st.number_input("Age",        0, 100, 25)

    with col2:
        room_service  = st.number_input("Room Service",   0.0, 15000.0, 0.0)
        food_court    = st.number_input("Food Court",     0.0, 15000.0, 0.0)
        shopping_mall = st.number_input("Shopping Mall",  0.0, 15000.0, 0.0)
        spa           = st.number_input("Spa",            0.0, 15000.0, 0.0)
        vr_deck       = st.number_input("VR Deck",        0.0, 15000.0, 0.0)
        cabin_num     = st.number_input("Cabin Number",   0, 2000, 0)
        group_size    = st.number_input("Group Size",     1, 20, 1)

    submitted = st.form_submit_button("Prediksi", type="primary", use_container_width=True)

if submitted:
    total = room_service + food_court + shopping_mall + spa + vr_deck
    input_df = pd.DataFrame([{
        "HomePlanet":   home_planet,
        "CryoSleep":    cryo_sleep,
        "Destination":  destination,
        "VIP":          vip,
        "Deck":         deck,
        "Side":         side,
        "Age":          float(age),
        "RoomService":  room_service,
        "FoodCourt":    food_court,
        "ShoppingMall": shopping_mall,
        "Spa":          spa,
        "VRDeck":       vr_deck,
        "Cabin_num":    float(cabin_num),
        "Group_size":   float(group_size),
        "Solo":         float(group_size == 1),
        "TotalSpending": total,
        "NoSpending":   float(total == 0),
    }])

    pred  = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    if pred == 1:
        st.success(f"**TRANSPORTED** — Probabilitas: {proba[1]*100:.1f}%")
    else:
        st.error(f"**NOT TRANSPORTED** — Probabilitas: {proba[0]*100:.1f}%")
