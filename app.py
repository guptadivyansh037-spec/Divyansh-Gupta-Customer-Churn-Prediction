import streamlit as st
import numpy as np
import joblib

# Page Configuration
st.set_page_config(page_title="Customer Churn Predictor", page_icon="🔮", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.write("Customer details fill kijiye aur check kijiye ki customer churn hoga ya nahi.")

# Load Model and Scaler
@st.cache_resource
def load_assets():
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    st.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.error("Model files load nahi ho paayi. Make sure 'churn_model.pkl' aur 'scaler.pkl' Google Colab me saved hain.")

st.markdown("---")

# Input Form
with st.form("churn_form"):
    st.subheader("📋 Customer Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", options=["Female", "Male"])
        senior_citizen = st.selectbox("Senior Citizen", options=["No", "Yes"])
        tenure = st.number_input("Tenure (Months)", min_value=1, max_value=100, value=12)
        contract = st.selectbox("Contract Type", options=["Month-to-month", "One year", "Two year"])
        
    with col2:
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=200.0, value=65.0)
        payment_method = st.selectbox("Payment Method", options=["Bank transfer", "Credit card", "Electronic check", "Mailed check"])
        
    # Calculate Total Charges automatically
    total_charges = tenure * monthly_charges
    st.info(f"Calculated Total Charges: **${total_charges:.2f}**")
    
    submit_button = st.form_submit_button(label="Predict Churn Risk")

# Prediction Logic
if submit_button:
    # Categorical variables encoding (matching training phase)
    gender_enc = 1 if gender == "Male" else 0
    senior_enc = 1 if senior_citizen == "Yes" else 0
    
    contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
    contract_enc = contract_map[contract]
    
    payment_map = {"Bank transfer": 0, "Credit card": 1, "Electronic check": 2, "Mailed check": 3}
    payment_enc = payment_map[payment_method]
    
    # Feature vector creation
    features = np.array([[gender_enc, senior_enc, tenure, monthly_charges, contract_enc, payment_enc, total_charges]])
    
    # Scale features
    scaled_features = scaler.transform(features)
    
    # Predict
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1] * 100
    
    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    
    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk!**\n\nIs customer ke left karne ke chances **{probability:.1f}%** hain.")
    else:
        st.success(f"✅ **Customer Likely to Stay.**\n\nChurn Probability only **{probability:.1f}%** hai.")
