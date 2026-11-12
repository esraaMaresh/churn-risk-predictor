import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Retention Intelligence", page_icon="🎯", layout="wide")

@st.cache_resource
def load_pipeline():
    return joblib.load("models/churn_pipeline.joblib")

pipeline = load_pipeline()

st.title("🎯 Customer Churn & Retention Intelligence")
st.markdown("Assess customer churn probability and explore risk factors in real-time.")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📋 Account & Service Profile")
    tenure = st.slider("Tenure (Months with company)", min_value=1, max_value=72, value=6)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=130.0, value=85.0)
    total_charges = st.number_input("Total Charges ($)", min_value=15.0, max_value=9000.0, value=float(tenure * monthly_charges))
    
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

with col2:
    st.subheader("🛡️ Add-ons & Demographics")
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

# Assemble input data matching the schema
input_dict = {
    'gender': [gender],
    'SeniorCitizen': [senior_citizen],
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'InternetService': [internet_service],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges]
}

input_df = pd.DataFrame(input_dict)

st.markdown("---")
if st.button("🚀 Evaluate Customer Risk", type="primary", use_container_width=True):
    prob = pipeline.predict_proba(input_df)[0][1]
    
    st.subheader("Evaluation Results")
    m1, m2 = st.columns([1, 2])
    
    with m1:
        st.metric(label="Predicted Churn Risk", value=f"{prob * 100:.1f}%")
        
    with m2:
        if prob >= 0.60:
            st.error("🚨 **High Churn Risk**: Customer is at immediate risk of leaving. High priority for proactive retention incentives (e.g. promotional discounts, long-term contract switch, dedicated support).")
        elif prob >= 0.35:
            st.warning("⚠️ **Moderate Churn Risk**: Customer shows sensitivity to pricing or service tier. Monitor engagement and satisfaction.")
        else:
            st.success("✅ **Low Churn Risk / Loyal Customer**: Customer demonstrates stable retention patterns.")