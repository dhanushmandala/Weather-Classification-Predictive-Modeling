import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load Saved Files
# ----------------------------
model = joblib.load("best_svm_weather_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
selector = joblib.load("selector.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Weather Classification", page_icon="🌦️")

st.title("🌦️ Weather Classification System")
st.write("Predict the weather type using the trained SVM model.")

# ----------------------------
# User Inputs
# ----------------------------
temperature = st.number_input("Temperature (°C)", value=25.0)
humidity = st.slider("Humidity (%)", 0, 100, 60)
wind_speed = st.number_input("Wind Speed (km/h)", value=10.0)
precipitation = st.slider("Precipitation (%)", 0, 100, 30)

cloud_cover = st.selectbox(
    "Cloud Cover",
    ["clear", "partly cloudy", "cloudy", "overcast"]
)

pressure = st.number_input(
    "Atmospheric Pressure (hPa)",
    value=1013.0
)

uv_index = st.slider("UV Index", 0, 15, 5)

season = st.selectbox(
    "Season",
    ["Spring", "Summer", "Autumn", "Winter"]
)

visibility = st.number_input(
    "Visibility (km)",
    value=10.0
)

location = st.selectbox(
    "Location",
    ["coastal", "inland", "mountain"]
)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Weather"):

    comfort_index = temperature * (1 - humidity / 100)

    data = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "Wind Speed": wind_speed,
        "Precipitation (%)": precipitation,
        "Cloud Cover": cloud_cover,
        "Atmospheric Pressure": pressure,
        "UV Index": uv_index,
        "Season": season,
        "Visibility (km)": visibility,
        "Location": location,
        "Comfort_Index": comfort_index
    }])

    # One-hot encoding
    data = pd.get_dummies(
        data,
        columns=["Cloud Cover", "Season", "Location"]
    )

    # Match training columns
    training_columns = selector.feature_names_in_

    data = data.reindex(
        columns=training_columns,
        fill_value=0
    )

    # Feature Selection
    data_selected = selector.transform(data)

    # Scaling
    data_scaled = scaler.transform(data_selected)

    # Prediction
    prediction = model.predict(data_scaled)

    weather = label_encoder.inverse_transform(prediction)

    st.success(f"### Predicted Weather: {weather[0]}")