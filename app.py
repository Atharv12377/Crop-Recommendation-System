import streamlit as st
import pickle
import pandas as pd

# 🌿 Page Configuration
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")

# 🎯 Load the trained model
try:
    with open('best_random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("⚠️ Model file 'best_random_forest_model.pkl' not found. Please make sure it's in the same directory.")
    st.stop()

# 🌾 App Title and Description
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #2e7d32;'>🌿 Crop Recommendation System 🌿</h1>
        <p style='color: #555; font-size: 18px;'>Enter your soil and weather details below to get the best crop suggestion for your farm.</p>
    </div>
""", unsafe_allow_html=True)

# 🌤️ Input Section
st.markdown("---")
st.subheader("🔢 Enter Environmental Parameters")

col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input('🧪 Nitrogen (N)', min_value=0.0, max_value=140.0, value=50.0)
    K = st.number_input('🧂 Potassium (K)', min_value=0.0, max_value=205.0, value=50.0)
    humidity = st.number_input('💧 Humidity (%)', min_value=0.0, max_value=100.0, value=50.0)

with col2:
    P = st.number_input('🧫 Phosphorus (P)', min_value=0.0, max_value=145.0, value=50.0)
    temperature = st.number_input('🌡️ Temperature (°C)', min_value=0.0, max_value=45.0, value=25.0)
    ph = st.number_input('⚗️ Soil pH', min_value=0.0, max_value=14.0, value=7.0)

with col3:
    rainfall = st.number_input('🌧️ Rainfall (mm)', min_value=0.0, max_value=300.0, value=100.0)
    st.write("")  # spacing

# 🌱 Predict Button
st.markdown("---")
if st.button('🚜 Get Crop Recommendation', use_container_width=True, type='primary'):
    # Create a DataFrame from user input
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    # Scale the input data using the loaded scaler
    input_data_scaled = loaded_scaler.transform(input_data)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data_scaled)

    # 🌾 Display Result (Professional Styled Card)
    st.markdown(f"""
        <div style="
            background-color:#f0fdf4;
            border:2px solid #81c784;
            border-radius:15px;
            padding:20px;
            text-align:center;
            margin-top:20px;
        ">
            <h2 style='color:#2e7d32;'>✅ Recommended Crop</h2>
            <p style='font-size:22px; color:#1b5e20; font-weight:bold;'>{prediction[0].capitalize()}</p>
            <p style='color:#4b5563;'>This crop is best suited for the provided soil and weather conditions.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <hr>
    <div style='text-align:center; color:grey; font-size:14px;'>
        Developed by <b>Group 8</b>.
    </div>
""", unsafe_allow_html=True)
