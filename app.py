import streamlit as st
import pickle
import pandas as pd

# Load the trained model
try:
    with open('best_random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'best_random_forest_model.pkl' not found. Please make sure it's in the same directory.")
    st.stop()

st.title('Crop Recommendation System')

st.write('Enter the environmental parameters to get a crop recommendation.')

# Create input fields for the features
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input('Nitrogen (N)', min_value=0.0, max_value=140.0, value=50.0)
    K = st.number_input('Potassium (K)', min_value=0.0, max_value=205.0, value=50.0)
    humidity = st.number_input('Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)

with col2:
    P = st.number_input('Phosphorus (P)', min_value=0.0, max_value=145.0, value=50.0)
    temperature = st.number_input('Temperature (°C)', min_value=0.0, max_value=45.0, value=25.0)
    ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.0)

with col3:
    rainfall = st.number_input('Rainfall (mm)', min_value=0.0, max_value=300.0, value=100.0)
    # Add an empty space to align the button
    st.write("") # Adjust spacing as needed


if st.button('Get Recommendation'):
    # Create a DataFrame from user input
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                               columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    # Make prediction
    prediction = model.predict(input_data)

    st.success(f'The recommended crop is: **{prediction[0]}**')