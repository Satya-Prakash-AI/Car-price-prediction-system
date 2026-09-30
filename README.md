# Car Price Prediction System

A machine learning-based web application that predicts the price of a car based on its features. The project uses data preprocessing, exploratory data analysis, feature encoding, and regression models to build an accurate car price prediction system.

## Features

- Exploratory Data Analysis (EDA)
- Data preprocessing and feature encoding
- Numerical feature scaling
- Multiple regression models
- Random Forest Regression
- XGBoost Regression
- Model evaluation using:
  - MAE
  - RMSE
  - R² Score
- Feature importance analysis
- Interactive Streamlit web application
- Saved trained XGBoost model using Joblib

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

## Dataset

The project uses a Ford car dataset containing features such as:

- Model
- Year
- Transmission
- Mileage
- Fuel Type
- Tax
- MPG
- Engine Size

The target variable is:

- Price

## Machine Learning Models

The project experiments with:

1. Linear Regression
2. Random Forest Regressor
3. XGBoost Regressor

The XGBoost model is saved as:

```text
car_price_model.pkl
