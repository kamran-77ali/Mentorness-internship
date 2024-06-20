import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load the trained model
with open('best_fraud_detection_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
scaler = StandardScaler()

# Define function to preprocess the input data
def preprocess_input(data):
    # Convert to DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    df_imputed = imputer.fit_transform(df)
    
    # Scale the features
    df_scaled = scaler.fit_transform(df_imputed)
    
    return df_scaled

# Streamlit UI
st.title("Real-time Fastag Fraud Detection")
st.write("Enter the details of the transaction below:")

# Input fields for transaction data
Transaction_Amount = st.number_input("Transaction Amount", value=0.0)
Amount_paid = st.number_input("Amount Paid", value=0.0)
Vehicle_Speed = st.number_input("Vehicle Speed", value=0.0)
Vehicle_Type = st.selectbox("Vehicle Type", ["Car", "Truck", "Bike", "Other"])
FastagID = st.text_input("Fastag ID")

# Additional features based on the above inputs
Amount_Ratio = Transaction_Amount / Amount_paid if Amount_paid != 0 else 0
Speed_Amount_Ratio = Vehicle_Speed / Transaction_Amount if Transaction_Amount != 0 else 0
Speed_X_Amount = Vehicle_Speed * Transaction_Amount

# Create a dictionary to hold the input values
input_data = {
    'Transaction_Amount': Transaction_Amount,
    'Amount_paid': Amount_paid,
    'Vehicle_Speed': Vehicle_Speed,
    'Vehicle_Type': Vehicle_Type,
    'FastagID': FastagID,
    'Amount_Ratio': Amount_Ratio,
    'Speed_Amount_Ratio': Speed_Amount_Ratio,
    'Speed_X_Amount': Speed_X_Amount,
    # Include other necessary features here
}

# When the "Predict" button is clicked
if st.button("Predict"):
    # Preprocess the input data
    input_df = pd.DataFrame([input_data])
    input_df = input_df.drop(columns=['Vehicle_Type', 'FastagID'])
    processed_data = preprocess_input(input_df)
    
    # Predict the probability of fraud
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)[0][1]
    
    # Display the prediction result
    if prediction == 1:
        st.error(f"The transaction is predicted to be fraudulent with a probability of {prediction_proba:.2f}")
    else:
        st.success(f"The transaction is predicted to be legitimate with a probability of {prediction_proba:.2f}")

