import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ----------------------------------------------------------
# 1️⃣ Load the trained CatBoost model
# ----------------------------------------------------------
with open("catboost_model.pkl", "rb") as file:
    model = pickle.load(file)

st.set_page_config(page_title="Used Car Price Prediction", layout="centered")

# ----------------------------------------------------------
# 2️⃣ Page Title and Description
# ----------------------------------------------------------
st.title("🚗 Used Car Price Prediction App")
st.markdown("""
This app predicts the **selling price of a used car** based on its specifications.  
Please fill in the details below to get an estimated price.
""")

# ----------------------------------------------------------
# 3️⃣ Input Fields
# ----------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Car Name (e.g., Maruti Swift)")
    year = st.number_input("Year of Purchase", min_value=1990, max_value=2025, value=2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, step=1000)
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])

with col2:
    seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
    owner = st.selectbox("Owner Type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
    mileage = st.number_input("Mileage (km/l)", min_value=5.0, max_value=50.0, value=18.0, step=0.1)
    engine = st.number_input("Engine (CC)", min_value=500, max_value=5000, value=1200, step=100)
    max_power = st.number_input("Max Power (bhp)", min_value=20.0, max_value=400.0, value=82.0, step=1.0)
    seats = st.selectbox("Number of Seats", [2, 4, 5, 6, 7, 8, 9])

# ----------------------------------------------------------
# 4️⃣ Prepare input data for prediction
# ----------------------------------------------------------
input_data = pd.DataFrame({
    "name": [name],
    "year": [year],
    "km_driven": [km_driven],
    "fuel": [fuel],
    "seller_type": [seller_type],
    "transmission": [transmission],
    "owner": [owner],
    "mileage": [mileage],
    "engine": [engine],
    "max_power": [max_power],
    "seats": [seats]
})

# ----------------------------------------------------------
# 5️⃣ Predict button
# ----------------------------------------------------------
if st.button("🔍 Predict Price"):
    try:
        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Selling Price: ₹ {prediction[0]:,.2f}")
    except Exception as e:
        st.error("⚠️ Error during prediction. Please check model or inputs.")
        st.write(e)

# ----------------------------------------------------------
# 6️⃣ Footer
# ----------------------------------------------------------
st.markdown("---")
st.caption("Developed by Sowjanya — Data Scientist | Used Car Price Prediction Project")
