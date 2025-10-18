import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# -------------------- Page Config --------------------
st.set_page_config(page_title="Car Sales Price Prediction", layout="wide")

# -------------------- Background Image --------------------
def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .stApp .block-container {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 2rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Example car image URL (replace with your own if needed)
set_bg_image("https://images.unsplash.com/photo-1603782276214-0c8f6f2db4a8?auto=format&fit=crop&w=1350&q=80")

# -------------------- Load Model and Scaler --------------------
with open('catboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# -------------------- Sidebar Inputs --------------------
st.sidebar.header("Enter Car Details")

car_year = st.sidebar.number_input("Car Year", min_value=2000, max_value=2025, value=2015)
km_driven = st.sidebar.number_input("Kilometers Driven (in km)", min_value=0, value=50000)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "LPG"])
dealer_type = st.sidebar.selectbox("Dealer Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Number of Owners", ["1st_Owner", "2nd_Owner", "3rd_Owner", "4th_Owner", "Test Drive Car"])

# -------------------- Prepare Input Data --------------------
car_age = 2025 - car_year

input_data = {
    'km_driven': km_driven,
    'car_age': car_age,
    'fuel_CNG': 1 if fuel_type == 'CNG' else 0,
    'fuel_Diesel': 1 if fuel_type == 'Diesel' else 0,
    'fuel_Electric': 1 if fuel_type == 'Electric' else 0,
    'fuel_LPG': 1 if fuel_type == 'LPG' else 0,
    'fuel_Petrol': 1 if fuel_type == 'Petrol' else 0,
    'seller_type_Dealer': 1 if dealer_type == 'Dealer' else 0,
    'seller_type_Individual': 1 if dealer_type == 'Individual' else 0,
    'seller_type_Trustmark Dealer': 1 if dealer_type == 'Trustmark Dealer' else 0,
    'transmission_Manual': 1 if transmission == 'Manual' else 0,
    'transmission_Automatic': 1 if transmission == 'Automatic' else 0,
    'owner_1st_Owner': 1 if owner == '1st_Owner' else 0,
    'owner_2nd_Owner': 1 if owner == '2nd_Owner' else 0,
    'owner_3rd_Owner': 1 if owner == '3rd_Owner' else 0,
    'owner_4th_Owner': 1 if owner == '4th_Owner' else 0,
    'owner_Test_Drive_Car': 1 if owner == 'Test Drive Car' else 0
}

input_df = pd.DataFrame([input_data])

# Scale numeric features
input_df[['km_driven', 'car_age']] = scaler.transform(input_df[['km_driven', 'car_age']])

# -------------------- Prediction --------------------
if st.sidebar.button("Predict Price"):
    predicted_price = model.predict(input_df)
    st.markdown(f"""
        <div style='background-color: #f0c14b; padding: 15px; border-radius: 10px; text-align: center'>
            <h2>Predicted Selling Price: ₹{predicted_price[0]:,.2f}</h2>
        </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------
# 6️⃣ Footer
# ----------------------------------------------------------
st.markdown("---")
st.caption("Developed by Sowjanya — Data Scientist | Used Car Price Prediction Project")






