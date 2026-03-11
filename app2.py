import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# Load model
# -----------------------------
model = load_model("lstm_model.h5")

# -----------------------------
# Load scaler
# -----------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("Air Quality Index Prediction (LSTM Model)")

st.write("Enter weather and pollution parameters to predict AQI")

# -----------------------------
# INPUT SECTION
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
aqi_lag2 = st.number_input("AQI Two Days Ago")

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict AQI"):

    input_features = np.array([[
        tavg,tmin,tmax,
        prcp,wspd,pres,
        humidity,
        pm25,pm10,o3,
        no2,so2,co,
        season,
        aqi_lag1,aqi_lag2
    ]])

    # create dummy row for scaler
    dummy = np.zeros((1,17))
    dummy[0,:-1] = input_features

    scaled = scaler.transform(dummy)

    X = scaled[:,:-1]

    # reshape for LSTM
    X = X.reshape((1,1,16))

    prediction = model.predict(X)

    dummy_pred = np.zeros((1,17))
    dummy_pred[0,-1] = prediction

    final_aqi = scaler.inverse_transform(dummy_pred)[0,-1]

    st.success(f"Predicted AQI: {round(final_aqi,2)}")

    # AQI Category
    if final_aqi <= 50:
        st.info("Air Quality: Good")
    elif final_aqi <= 100:
        st.info("Air Quality: Moderate")
    elif final_aqi <= 150:
        st.warning("Air Quality: Unhealthy for Sensitive Groups")
    elif final_aqi <= 200:
        st.warning("Air Quality: Unhealthy")
    elif final_aqi <= 300:
        st.error("Air Quality: Very Unhealthy")
    else:
        st.error("Air Quality: Hazardous")
