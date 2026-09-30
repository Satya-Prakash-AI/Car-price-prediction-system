import streamlit as st
import pandas as pd
import joblib
# LOAD TRAINED MODEL

model = joblib.load("car_price_model.pkl")

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Ford Car Price Prediction",
    layout="wide"
)

# TITLE

st.title("Ford Used Car Price Prediction")

st.write(
    "Enter the car details below to predict its estimated selling price."
)

# SIDEBAR - USER INPUTS

st.sidebar.header("Car Details")

# Model

car_model = st.sidebar.selectbox(
    "Car Model",
    [
        "C-MAX",
        "EcoSport",
        "Edge",
        "Escort",
        "Fiesta",
        "Focus",
        "Fusion",
        "Galaxy",
        "Grand C-MAX",
        "Grand Tourneo Connect",
        "KA",
        "Ka+",
        "Kuga",
        "Mondeo",
        "Mustang",
        "Puma",
        "Ranger",
        "S-MAX",
        "Streetka",
        "Tourneo Connect",
        "Tourneo Custom",
        "Transit Tourneo",
        "Focus"
    ]
)

# Year

year = st.sidebar.number_input(
    "Year",
    min_value=1990,
    max_value=2026,
    value=2018,
    step=1
)

# Mileage

mileage = st.sidebar.number_input(
    "Mileage",
    min_value=0,
    value=45000,
    step=1000
)

# Transmission

transmission = st.sidebar.selectbox(
    "Transmission",
    [
        "Manual",
        "Semi-Auto",
        "Automatic"
    ]
)

# Fuel Type

fuel_type = st.sidebar.selectbox(
    "Fuel Type",
    [
        "Electric",
        "Hybrid",
        "Other",
        "Petrol",
        "Diesel"
    ]
)

# Tax

tax = st.sidebar.number_input(
    "Tax",
    min_value=0,
    value=150,
    step=5
)

# MPG

mpg = st.sidebar.number_input(
    "MPG",
    min_value=0.0,
    value=50.0,
    step=0.1
)

# Engine Size

engine_size = st.sidebar.number_input(
    "Engine Size",
    min_value=0.0,
    value=1.5,
    step=0.1
)

# PREDICTION

if st.button("Predict Selling Price"):

    # Create dataframe containing numerical features

    input_data = {
        "year": year,
        "mileage": mileage,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engine_size
    }

    # Model One-Hot Encoding

    model_columns = [
        "model_ C-MAX",
        "model_ EcoSport",
        "model_ Edge",
        "model_ Escort",
        "model_ Fiesta",
        "model_ Focus",
        "model_ Fusion",
        "model_ Galaxy",
        "model_ Grand C-MAX",
        "model_ Grand Tourneo Connect",
        "model_ KA",
        "model_ Ka+",
        "model_ Kuga",
        "model_ Mondeo",
        "model_ Mustang",
        "model_ Puma",
        "model_ Ranger",
        "model_ S-MAX",
        "model_ Streetka",
        "model_ Tourneo Connect",
        "model_ Tourneo Custom",
        "model_ Transit Tourneo",
        "model_Focus"
    ]


    for column in model_columns:
        input_data[column] = 0


    # Activate selected model column
    selected_model_column = "model_ " + car_model

    if selected_model_column in input_data:
        input_data[selected_model_column] = 1


    # Special case for Focus
    if car_model == "Focus":
        if "model_Focus" in input_data:
            input_data["model_Focus"] = 1

    # Transmission One-Hot Encoding

    input_data["transmission_Manual"] = 0
    input_data["transmission_Semi-Auto"] = 0

    if transmission == "Manual":
        input_data["transmission_Manual"] = 1

    elif transmission == "Semi-Auto":
        input_data["transmission_Semi-Auto"] = 1

    # Fuel Type One-Hot Encoding

    input_data["fuelType_Electric"] = 0
    input_data["fuelType_Hybrid"] = 0
    input_data["fuelType_Other"] = 0
    input_data["fuelType_Petrol"] = 0


    if fuel_type == "Electric":
        input_data["fuelType_Electric"] = 1

    elif fuel_type == "Hybrid":
        input_data["fuelType_Hybrid"] = 1

    elif fuel_type == "Other":
        input_data["fuelType_Other"] = 1

    elif fuel_type == "Petrol":
        input_data["fuelType_Petrol"] = 1

    new_car = pd.DataFrame([input_data])

    # Arrange columns in exactly the same order
    # expected by the trained model

    expected_columns = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize",
        "model_ C-MAX",
        "model_ EcoSport",
        "model_ Edge",
        "model_ Escort",
        "model_ Fiesta",
        "model_ Focus",
        "model_ Fusion",
        "model_ Galaxy",
        "model_ Grand C-MAX",
        "model_ Grand Tourneo Connect",
        "model_ KA",
        "model_ Ka+",
        "model_ Kuga",
        "model_ Mondeo",
        "model_ Mustang",
        "model_ Puma",
        "model_ Ranger",
        "model_ S-MAX",
        "model_ Streetka",
        "model_ Tourneo Connect",
        "model_ Tourneo Custom",
        "model_ Transit Tourneo",
        "model_Focus",
        "transmission_Manual",
        "transmission_Semi-Auto",
        "fuelType_Electric",
        "fuelType_Hybrid",
        "fuelType_Other",
        "fuelType_Petrol"
    ]


    new_car = new_car[expected_columns]

    # MAKE PREDICTION

    prediction = model.predict(new_car)[0]

    # DISPLAY RESULT

    st.success(
        f"Estimated Selling Price: £{prediction:,.0f}"
    )

    # DISPLAY INPUT DETAILS

    st.subheader("Entered Car Details")

    display_data = pd.DataFrame({
        "Feature": [
            "Model",
            "Year",
            "Mileage",
            "Transmission",
            "Fuel Type",
            "Tax",
            "MPG",
            "Engine Size"
        ],
        "Value": [
            car_model,
            year,
            mileage,
            transmission,
            fuel_type,
            tax,
            mpg,
            engine_size
        ]
    })

    st.table(display_data)