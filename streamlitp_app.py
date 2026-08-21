import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈️",
    layout="centered"
)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("model/flight_fare_model.pkl")


model = load_model()


# --------------------------------------------------
# App title
# --------------------------------------------------

st.title("✈️ Flight Fare Prediction")
st.write("Enter the flight details below to predict the ticket price.")


# --------------------------------------------------
# User inputs
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:
    airline = st.selectbox(
        "Airline",
        [
            "Jet Airways",
            "IndiGo",
            "Air India",
            "Multiple carriers",
            "SpiceJet",
            "Vistara",
            "GoAir",
            "Multiple carriers Premium economy",
            "Jet Airways Business",
            "Vistara Premium economy",
            "Trujet"
        ]
    )

    source = st.selectbox(
        "Source",
        [
            "Banglore",
            "Kolkata",
            "Delhi",
            "Chennai",
            "Mumbai"
        ]
    )

    destination = st.selectbox(
        "Destination",
        [
            "New Delhi",
            "Banglore",
            "Cochin",
            "Kolkata",
            "Delhi",
            "Hyderabad"
        ]
    )

    total_stops = st.selectbox(
        "Total Stops",
        [
            0,
            1,
            2,
            3,
            4
        ]
    )

with col2:
    journey_day = st.number_input(
        "Journey Day",
        min_value=1,
        max_value=31,
        value=15
    )

    journey_month = st.number_input(
        "Journey Month",
        min_value=1,
        max_value=12,
        value=5
    )

    journey_year = st.number_input(
        "Journey Year",
        min_value=2018,
        max_value=2030,
        value=2019
    )

    duration_minutes = st.number_input(
        "Duration (minutes)",
        min_value=1,
        max_value=2000,
        value=180
    )


# --------------------------------------------------
# Departure time
# --------------------------------------------------

st.subheader("Departure Time")

col1, col2 = st.columns(2)

with col1:
    departure_hour = st.number_input(
        "Departure Hour",
        min_value=0,
        max_value=23,
        value=10
    )

with col2:
    departure_minute = st.number_input(
        "Departure Minute",
        min_value=0,
        max_value=59,
        value=30
    )


# --------------------------------------------------
# Arrival time
# --------------------------------------------------

st.subheader("Arrival Time")

col1, col2 = st.columns(2)

with col1:
    arrival_hour = st.number_input(
        "Arrival Hour",
        min_value=0,
        max_value=23,
        value=13
    )

with col2:
    arrival_minute = st.number_input(
        "Arrival Minute",
        min_value=0,
        max_value=59,
        value=30
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Flight Fare", type="primary"):

    input_data = pd.DataFrame({
        "Airline": [airline],
        "Source": [source],
        "Destination": [destination],
        "Total_Stops": [total_stops],
        "Journey_Day": [journey_day],
        "Journey_Month": [journey_month],
        "Journey_Year": [journey_year],
        "Departure_Hour": [departure_hour],
        "Departure_Minute": [departure_minute],
        "Arrival_Hour": [arrival_hour],
        "Arrival_Minute": [arrival_minute],
        "Duration_Minutes": [duration_minutes]
    })

    try:
        prediction = model.predict(input_data)

        predicted_price = prediction[0]

        st.success(
            f"### Predicted Flight Fare: ₹{predicted_price:,.2f}"
        )

    except Exception as e:
        st.error("Unable to make the prediction.")
        st.exception(e)


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")
st.caption("Flight Fare Prediction ML Project")

