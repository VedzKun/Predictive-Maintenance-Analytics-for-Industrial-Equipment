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
st.title("Machine Status Predictor 🔮")
st.markdown("""
    This application predicts the status of a machine based on its operational parameters,
    advising whether it is safe to use or if it requires immediate service.
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
if st.sidebar.button("Predict Status"):
    # 1. Preprocess user inputs to match the model's training data format
    
    # Handle one-hot encoding for the 'Type' feature
    type_L, type_M, type_H = (0, 0, 0)
    if product_type == 'L':
        type_L = 1
    elif product_type == 'M':
        type_M = 1
    else:
        type_H = 1
        
    # Engineer new features exactly as they were created in the training notebook
    # Corrected feature names to match the model's expectations from the error log
    temp_diff = process_temp - air_temp
    power = torque * rotational_speed
    
    # Corrected failure flag logic to match the model's expectations from the error log
    # Assuming 'Overheat_Flag' was a simple binary flag based on temperature difference.
    # The threshold of 8.6K is from the dataset's documentation for HDF.
    overheat_flag = 1 if temp_diff < 8.6 else 0

    # Create the input DataFrame
    # Corrected column names and included 'Type_L' as expected by the model
    input_data = pd.DataFrame({
        'Air temperature [K]': [air_temp],
        'Process temperature [K]': [process_temp],
        'Rotational speed [rpm]': [rotational_speed],
        'Torque [Nm]': [torque],
        'Tool wear [min]': [tool_wear],
        'Type_L': [type_L],
        'Type_M': [type_M],
        'Temp_diff': [temp_diff],
        'Power': [power],
        'Overheat_Flag': [overheat_flag]
    })
    
    # 2. Make the prediction
    predicted_rul = model.predict(input_data)[0]

    # 3. Display the result
    st.subheader("Machine Status")

    # Define a simple threshold for RUL
    service_threshold = 20 # Inferred from the dataset's scale

    # If predicted_rul is numeric, compare as before
    try:
        rul_value = float(predicted_rul)
        if rul_value <= service_threshold:
            st.error("Status: **Needs Service**")
            st.markdown(f"The model predicts the machine has an estimated Remaining Useful Life of **{max(0, int(rul_value))}** time units.")
        else:
            st.success("Status: **Good to Use**")
            st.markdown(f"The model predicts the machine has an estimated Remaining Useful Life of **{int(rul_value)}** time units.")
    except ValueError:
        # If predicted_rul is a string label
        st.info(f"Predicted status: **{predicted_rul}**")

    st.markdown("---")
    st.info("This prediction is an estimate based on the current operational data. Regular monitoring is recommended.")

# Display the features expected by the model
st.write("Model expects features:", model.feature_names_in_)
