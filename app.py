import pickle
import pandas as pd

# Load the fitted scaler
scaler_filename = 'scaler.pkl'
loaded_scaler = pickle.load(open(scaler_filename, 'rb'))

# Load the trained model (assuming you saved your best model as 'best_random_forest_model.pkl')
model_filename = 'best_random_forest_model.pkl'
loaded_model = pickle.load(open(model_filename, 'rb'))

def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Predicts the best crop based on input features.

    Args:
        N (float): Nitrogen content in the soil.
        P (float): Phosphorus content in the soil.
        K (float): Potassium content in the soil.
        temperature (float): Temperature in Celsius.
        humidity (float): Humidity percentage.
        ph (float): pH value of the soil.
        rainfall (float): Rainfall in mm.

    Returns:
        str: The predicted crop label.
    """
    # Create a pandas DataFrame from the input values
    input_data = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]],
                              columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

    # Scale the input data using the loaded scaler
    input_data_scaled = loaded_scaler.transform(input_data)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(input_data_scaled)

    return prediction[0]

# Example usage (you would replace this with your Streamlit input handling)
# sample_input = predict_crop(90, 42, 43, 20.88, 82.00, 6.50, 202.94)
# print(f"Predicted crop: {sample_input}")