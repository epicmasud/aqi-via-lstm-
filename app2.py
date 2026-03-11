import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

st.title("AQI Prediction using LSTM")

# Load model
model = load_model("lstm_model.h5")

# Load scaler
scaler = pickle.load(open("scaler.pkl","rb"))

st.header("Input Features")

features = []

tavg = st.number_input("tavg")
features.append(tavg)

tmin = st.number_input("tmin")
features.append(tmin)

tmax = st.number_input("tmax")
features.append(tmax)

prcp = st.number_input("prcp")
features.append(prcp)

wspd = st.number_input("wspd")
features.append(wspd)

pres = st.number_input("pres")
features.append(pres)

Temperature = st.number_input("Temperature")
features.append(Temperature)

Humidity = st.number_input("Humidity")
features.append(Humidity)

Wind_Speed = st.number_input("Wind_Speed")
features.append(Wind_Speed)

Pressure = st.number_input("Pressure")
features.append(Pressure)

aqi_lag1 = st.number_input("AQI_lag1")
features.append(aqi_lag1)

season = st.number_input("season")
features.append(season)

PM25 = st.number_input("PM2.5")
features.append(PM25)

PM10 = st.number_input("PM10")
features.append(PM10)

O3 = st.number_input("O3")
features.append(O3)

NO2 = st.number_input("NO2")
features.append(NO2)

SO2 = st.number_input("SO2")
features.append(SO2)

CO = st.number_input("CO")
features.append(CO)

if st.button("Predict"):

    data = np.array(features).reshape(1,-1)

    data_scaled = scaler.transform(data)

    data_scaled = data_scaled.reshape(1,1,data_scaled.shape[1])

    prediction = model.predict(data_scaled)

    st.success(f"Predicted AQI: {prediction[0][0]}")
