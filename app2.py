import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# Load model and scaler
# -----------------------------
model = load_model("aqi_lstm_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# -----------------------------
# App Title
# -----------------------------
st.title("AQI Prediction using LSTM")

st.write("Enter Weather and Pollution Parameters")

# -----------------------------
# Input features
# -----------------------------

tavg = st.number_input("Average Temperature")
tmin = st.number_input("Minimum Temperature")
tmax = st.number_input("Maximum Temperature")

prcp = st.number_input("Precipitation")
wspd = st.number_input("Wind Speed")
pres = st.number_input("Pressure")

humidity = st.number_input("Humidity")

pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
o3 = st.number_input("O3")

no2 = st.number_input("NO2")
so2 = st.number_input("SO2")
co = st.number_input("CO")

season = st.selectbox("Season", [1,2,3,4])

aqi_lag1 = st.number_input("Previous Day AQI")
aqi_lag2 = st.number_input("Two Days Ago AQI")

# -----------------------------
# Prediction Button
# -----------------------------

if st.button("Predict AQI"):

    features = np.array([[

        tavg,tmin,tmax,prcp,wspd,pres,
        humidity,
        pm25,pm10,o3,no2,so2,co,
        season,aqi_lag1,aqi_lag2

    ]])

    # Dummy AQI column for scaler
    dummy = np.zeros((1, len(features[0]) + 1))
    dummy[0,:-1] = features

    scaled = scaler.transform(dummy)

    X = scaled[:,:-1]

    # reshape for LSTM
    X = X.reshape((1,1,X.shape[1]))

    pred = model.predict(X)

    dummy_pred = np.zeros((1, len(features[0]) + 1))
    dummy_pred[0,-1] = pred

    predicted_aqi = scaler.inverse_transform(dummy_pred)[0,-1]

    st.success(f"Predicted AQI: {predicted_aqi:.2f}")
