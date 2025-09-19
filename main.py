import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set the page title and layout
st.set_page_config(page_title="Machine RUL Predictor", layout="wide")

# --- Load the trained model ---
@st.cache_resource
def load_model():
    """Loads the pre-trained model from the file."""
    try:
        model = joblib.load('predictive_maintenance_model.joblib')
        return model
    except FileNotFoundError:
        st.error("Model file 'predictive_maintenance_model.joblib' not found. Please ensure it's in the same directory.")
        st.stop()

model = load_model()

# --- App UI ---
st.title("Machine Remaining Useful Life (RUL) Predictor 🔮")
st.markdown("""
    This application predicts the **Remaining Useful Life (RUL)** of a machine based on its current operational parameters.
    The model was trained on a predictive maintenance dataset to forecast the time remaining until a potential failure.
""")

st.markdown("---")

# --- Sidebar for User Input ---
st.sidebar.header("Machine Operational Data")

# Create input widgets for each feature.
# These must match the columns used to train the model, including engineered features.
# The min, max, and default values are based on the dataset's characteristics.

air_temp = st.sidebar.slider("Air Temperature [K]", min_value=295.0, max_value=305.0, value=298.5, step=0.1)
process_temp = st.sidebar.slider("Process Temperature [K]", min_value=305.0, max_value=315.0, value=310.0, step=0.1)
rotational_speed = st.sidebar.slider("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500, step=10)
torque = st.sidebar.slider("Torque [Nm]", min_value=0.0, max_value=80.0, value=40.0, step=0.5)
tool_wear = st.sidebar.slider("Tool Wear [min]", min_value=0, max_value=250, value=100, step=1)
product_type = st.sidebar.selectbox("Product Type", options=['L', 'M', 'H'])

# --- Prediction Button ---
if st.sidebar.button("Predict RUL"):
    # 1. Preprocess user inputs to match the model's training data format
    
    # Handle one-hot encoding for the 'Type' feature
    type_L, type_M, type_H = (0, 0, 0)
    if product_type == 'L':
        type_L = 1
    elif product_type == 'M':
        type_M = 1
    else:
        type_H = 1
        
    # Assume 'Type_L' was dropped in one-hot encoding during training
    # This ensures consistency
    
    # Engineer new features exactly as they were created in the training notebook
    temp_diff = process_temp - air_temp
    power_w = torque * rotational_speed
    overstrain_flag = 1 if torque > 60 else 0

    # Create the input DataFrame
    # Column order is crucial, so match it exactly
    input_data = pd.DataFrame({
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rotational_speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear],
        'Type_M': [type_M],
        'Type_H': [type_H],
        'Temp_Diff': [temp_diff],
        'Power [W]': [power_w],
        'Overstrain_Flag': [overstrain_flag]
    })
    
    # 2. Make the prediction
    prediction = model.predict(input_data)[0]
    
    # 3. Display the result
    st.subheader("Predicted Remaining Useful Life (RUL)")
    
    # Round the prediction for a cleaner display
    predicted_rul = int(round(prediction))

    if predicted_rul < 0:
        st.warning(f"The model predicts an imminent failure. RUL: **{max(0, predicted_rul)}** time units.")
    else:
        st.success(f"Predicted RUL: **{predicted_rul}** time units.")
        
    st.markdown("---")
    st.info("This prediction is an estimate based on the current operational data. Regular monitoring is recommended.")