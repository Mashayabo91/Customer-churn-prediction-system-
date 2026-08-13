import streamlit as st
import pandas as pd
import joblib

# Load the complete saved pipeline:
# preprocessing + trained machine-learning model.
model = joblib.load("churn_model.joblib")

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Enter customer information to estimate the probability "
    "that the customer will leave the bank."
)

st.divider()

credit_score = st.number_input(
    "Credit Score", min_value=300, max_value=850, value=650, step=1
)

geography = st.selectbox(
    "Geography", ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender", ["Female", "Male"]
)

age = st.number_input(
    "Age", min_value=18, max_value=100, value=40, step=1
)

tenure = st.number_input(
    "Tenure (years)", min_value=0, max_value=10, value=5, step=1
)

balance = st.number_input(
    "Account Balance", min_value=0.0, value=50000.0, step=1000.0
)

num_products = st.number_input(
    "Number of Bank Products", min_value=1, max_value=4, value=2, step=1
)

has_card = st.selectbox(
    "Has Credit Card?", ["Yes", "No"]
)

is_active = st.selectbox(
    "Is Active Member?", ["Yes", "No"]
)

estimated_salary = st.number_input(
    "Estimated Salary", min_value=0.0, value=100000.0, step=1000.0
)

if st.button("Predict Churn", type="primary", use_container_width=True):

    # Build one-row DataFrame using exactly the feature names
    # and structure expected by the saved model.
    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": 1 if is_active == "Yes" else 0,
        "EstimatedSalary": estimated_salary
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0, 1]

    st.divider()

    if prediction == 1:
        st.error("⚠️ The customer is predicted to be at risk of churn.")
    else:
        st.success("✅ The customer is predicted to stay.")

    st.metric(
        "Estimated Churn Probability",
        f"{probability:.1%}"
    )

    st.progress(float(probability))

    st.caption(
        "This prediction is a machine-learning estimate and should "
        "support, not replace, business judgment."
    )
