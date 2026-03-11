import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Page config
st.set_page_config(page_title="AQI Prediction using LSTM", layout="centered")

st.title("🌫️ Air Quality Index Prediction")
st.write("Predict AQI using LSTM Model")

# Load model
model = load_model("lstm_model.h5")

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.subheader("Enter Environmental Data")

# User inputs
tavg = st.number_input("Average Temperature")
tmin = st.number_input("Minimum Temperature")
tmax = st.number_input("Maximum Temperature")
prcp = st.number_input("Precipitation")
wspd = st.number_input("Wind Speed")
pres = st.number_input("Pressure")

Temperature = st.number_input("Temperature")
Humidity = st.number_input("Humidity")
Wind_Speed = st.number_input("Wind Speed (Alt)")
Pressure = st.number_input("Pressure (Alt)")

aqi_lag1 = st.number_input("Previous Day AQI")

season = st.selectbox(
    "Season",
    [1,2,3,4]
)

PM25 = st.number_input("PM2.5")
PM10 = st.number_input("PM10")
O3 = st.number_input("O3")
NO2 = st.number_input("NO2")
SO2 = st.number_input("SO2")
CO = st.number_input("CO")

# Feature list
features = [
    tavg,tmin,tmax,prcp,wspd,pres,
    Temperature,Humidity,Wind_Speed,Pressure,
    aqi_lag1,season,
    PM25,PM10,O3,NO2,SO2,CO
]

# Predict button
if st.button("Predict AQI"):

    data = np.array(features).reshape(1,-1)

    # Scale data
    data_scaled = scaler.transform(data)

    # LSTM expects 3D input
    data_scaled = data_scaled.reshape(1,1,data_scaled.shape[1])

    prediction = model.predict(data_scaled)

    aqi_value = prediction[0][0]

    st.success(f"Predicted AQI: {aqi_value:.2f}")

    # AQI Category
    if aqi_value <= 50:
        st.info("Good Air Quality")
    elif aqi_value <= 100:
        st.info("Moderate Air Quality")
    elif aqi_value <= 150:
        st.warning("Unhealthy for Sensitive Groups")
    elif aqi_value <= 200:
        st.warning("Unhealthy")
    elif aqi_value <= 300:
        st.error("Very Unhealthy")
    else:
        st.error("Hazardous")
