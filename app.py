import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the pre-trained CatBoost model
model = pickle.load(open('catboost_model.pkl', 'rb'))

# Streamlit app layout
st.title("Car Sales Price Prediction")

# Collect user input for prediction
st.sidebar.header("Enter Car Details")

# Common popular car names
car_names = [
    "Maruti Swift", "Maruti Alto", "Maruti Baleno", "Hyundai i10", "Hyundai i20",
    "Hyundai Creta", "Honda City", "Honda Amaze", "Tata Nexon", "Tata Tiago",
    "Kia Seltos", "Mahindra Scorpio", "Mahindra XUV500", "Toyota Innova",
    "Toyota Fortuner", "Renault Kwid", "Skoda Rapid", "Volkswagen Polo",
    "Ford EcoSport", "Nissan Magnite"
]

# User inputs for prediction
car_name = st.sidebar.selectbox("Select Car Name", car_names)
car_year = st.sidebar.number_input("Car Year", min_value=2000, max_value=2025, value=2015)
km_driven = st.sidebar.number_input("Kilometers Driven (in km)", min_value=0, value=50000)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "LPG"])
dealer_type = st.sidebar.selectbox("Dealer Type", ["Dealer", "Individual", "Trustmark Dealer"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("Number of Owners", ["1st_Owner", "2nd_Owner", "3rd_Owner", "4th_Owner", "Test Drive Car"])

# Calculate car age
car_age = 2025 - car_year

# Prepare the input data
input_data = {
    'km_driven': km_driven,
    'fuel_CNG': 1 if fuel_type == 'CNG' else 0,    
    'fuel_Diesel': 1 if fuel_type == 'Diesel' else 0,    
    'fuel_Electric': 1 if fuel_type == 'Electric' else 0,
    'fuel_LPG': 1 if fuel_type == 'LPG' else 0,
    'fuel_Petrol': 1 if fuel_type == 'Petrol' else 0,
    'seller_type_Dealer': 1 if dealer_type == 'Dealer' else 0,  
    'seller_type_Individual': 1 if dealer_type == 'Individual' else 0,  
    'seller_type_Trustmark Dealer': 1 if dealer_type == 'Trustmark Dealer' else 0,  
    'transmission_Automatic': 1 if transmission == 'Automatic' else 0,
    'transmission_Manual': 1 if transmission == 'Manual' else 0,
    'owner_4th_Owner': 1 if owner == '4th_Owner' else 0,
    'owner_First Owner': 1 if owner == '1st_Owner' else 0,
    'owner_Second Owner': 1 if owner == '2nd_Owner' else 0,
    'owner_Test Drive Car': 1 if owner == 'Test Drive Car' else 0,
    'owner_Third Owner': 1 if owner == '3rd_Owner' else 0,
    'car_age': car_age
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Standardize numerical features
scaler = StandardScaler()
input_df[['km_driven', 'car_age']] = scaler.fit_transform(input_df[['km_driven', 'car_age']])

# Predict price
if st.sidebar.button("Predict Price"):
    predicted_price = model.predict(input_df)
    
    st.subheader("Prediction Result")
    st.write(f"**Car Name:** {car_name}")
    st.write(f"**Predicted Selling Price:** ₹{predicted_price[0]:,.2f}")
    st.success(f"✅ The estimated price for your {car_name} is ₹{predicted_price[0]:,.2f}")

# Footer
st.markdown("---")
st.caption("Developed by Sowjanya — Data Scientist | Used Car Price Prediction Project")

