import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------------------
# Load Saved Model & Scaler
# ---------------------------
model = load_model("customer_churn_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("🔮 Customer Churn Prediction App")
st.write("Provide customer details below to predict whether the customer will exit the bank.")

# ---------------------------
# Input Fields
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
    Age = st.number_input("Age", min_value=18, max_value=100, step=1)
    Tenure = st.number_input("Tenure (years)", min_value=0, max_value=10)

with col2:
    Balance = st.number_input("Balance Amount (₹)", min_value=0.0, step=100.0)
    NumOfProducts = st.selectbox("Number of Products", [1, 2, 3, 4])
    HasCrCard = st.selectbox("Has Credit Card?", ["Yes", "No"])

with col3:
    IsActiveMember = st.selectbox("Active Member?", ["Yes", "No"])
    EstimatedSalary = st.number_input("Estimated Salary (₹)", min_value=0.0, step=1000.0)
    Gender = st.selectbox("Gender", ["Male", "Female"])

Geography = st.selectbox("Geography", ["France", "Spain", "Germany"])


# ---------------------------
# Convert input to model format
# ---------------------------

# One-hot encoding manually (because your dataset has these)
Gender_Male = 1 if Gender == "Male" else 0

Geography_France = 1 if Geography == "France" else 0
Geography_Germany = 1 if Geography == "Germany" else 0
Geography_Spain = 1 if Geography == "Spain" else 0

HasCrCard = 1 if HasCrCard == "Yes" else 0
IsActiveMember = 1 if IsActiveMember == "Yes" else 0

# Prepare feature vector (same order as training)
input_data = np.array([[CreditScore, Age, Tenure, Balance, NumOfProducts,
                        HasCrCard, IsActiveMember, EstimatedSalary,
                        Gender_Male, Geography_Germany, Geography_Spain]])

# Scale input
input_scaled = scaler.transform(input_data)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0][0]
    result = "❌ Customer Will Leave the Bank" if prediction > 0.5 else "✅ Customer Will Stay"

    st.subheader("Prediction Result")
    st.write(result)

    st.metric("Churn Probability", f"{prediction*100:.2f}%")
