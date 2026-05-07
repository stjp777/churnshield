import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="ChurnShield",
    layout="wide"
)

# load model and columns
model = joblib.load("churn_model.pkl")
columns = joblib.load("model_columns.pkl")

@st.cache_data
def load_reference_data():
    return pd.read_csv("churn_clean.csv")

# app title
st.title("ChurnShield")
st.write("Predict whether a customer is likely to cancel their subscription.")

st.sidebar.header("Customer details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly charges ($)", 20, 120, 65)
total_charges = tenure * monthly_charges

contract = st.sidebar.selectbox(
    "Contract type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet service",
    ["DSL", "Fiber optic", "No"]
)

tech_support = st.sidebar.selectbox(
    "Tech support",
    ["Yes", "No", "No internet service"]
)

senior_citizen = st.sidebar.checkbox("Senior citizen")
partner = st.sidebar.checkbox("Has partner")
dependents = st.sidebar.checkbox("Has dependents")
paperless_billing = st.sidebar.checkbox("Paperless billing")

online_security = st.sidebar.selectbox(
    "Online security",
    ["Yes", "No", "No internet service"]
)

payment_method = st.sidebar.selectbox(
    "Payment method",
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"]
)

st.divider()

if st.button("Predict churn risk"):

    # 1 — start with all zeros
    input_data = pd.DataFrame([np.zeros(len(columns))], columns=columns)

    # 2 — fill numeric columns directly
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges
    input_data["SeniorCitizen"] = int(senior_citizen)

    # 3 — encode categorical selections
    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1

    if internet_service == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    elif internet_service == "No":
        input_data["InternetService_No"] = 1

    if tech_support == "Yes":
        input_data["TechSupport_Yes"] = 1
    elif tech_support == "No internet service":
        input_data["TechSupport_No internet service"] = 1

    if online_security == "Yes":
        input_data["OnlineSecurity_Yes"] = 1
    elif online_security == "No internet service":
        input_data["OnlineSecurity_No internet service"] = 1

    if payment_method == "Credit card (automatic)":
        input_data["PaymentMethod_Credit card (automatic)"] = 1
    elif payment_method == "Electronic check":
        input_data["PaymentMethod_Electronic check"] = 1
    elif payment_method == "Mailed check":
        input_data["PaymentMethod_Mailed check"] = 1

    if partner:
        input_data["Partner_Yes"] = 1
    if dependents:
        input_data["Dependents_Yes"] = 1
    if paperless_billing:
        input_data["PaperlessBilling_Yes"] = 1

    # 4 — predict
    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    # 5 — display result
    st.subheader("Result")

    if probability >= 0.7:
        st.error(f"High churn risk — {probability:.1%} probability")
    elif probability >= 0.4:
        st.warning(f"Medium churn risk — {probability:.1%} probability")
    else:
        st.success(f"Low churn risk — {probability:.1%} probability")

    st.progress(float(probability))

    st.subheader("Recommended actions")

    recommendations = []

    if contract == "Month-to-month":
        recommendations.append("Offer a discounted annual contract")
    if payment_method == "Electronic check":
        recommendations.append("Incentivize switch to automatic payment")
    if tech_support == "No":
        recommendations.append("Offer a free tech support trial")
    if online_security == "No":
        recommendations.append("Offer online security add-on")
    if tenure < 12:
        recommendations.append("Assign a customer success rep — early tenure is high risk")
    if monthly_charges > 75:
        recommendations.append("Review plan pricing — charge is above average")

    if recommendations:
        for rec in recommendations:
            st.write(f"• {rec}")
    else:
        st.write("No immediate actions needed — customer profile looks stable.")

    st.subheader("Risk factors detected")

    risk_factors = []

    if tenure < 12:
        risk_factors.append(("Short tenure", f"{tenure} months"))
    if monthly_charges > 75:
        risk_factors.append(("High monthly charge", f"${monthly_charges}"))
    if contract == "Month-to-month":
        risk_factors.append(("Month-to-month contract", "No long-term commitment"))
    if payment_method == "Electronic check":
        risk_factors.append(("Electronic check payment", "Non-automatic billing"))
    if tech_support == "No":
        risk_factors.append(("No tech support", "Linked to higher churn"))
    if internet_service == "Fiber optic":
        risk_factors.append(("Fiber optic service", "Higher churn segment"))

    if risk_factors:
        for factor, detail in risk_factors:
            st.write(f"**{factor}** — {detail}")
    else:
        st.write("No major risk factors detected.")

    st.subheader("How this customer compares")
    df_ref = load_reference_data()

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    # Tenure comparison
    axes[0].hist(df_ref["tenure"], bins=30, color="steelblue", alpha=0.6)
    axes[0].axvline(tenure, color="tomato", linewidth=2, label="This customer")
    axes[0].set_title("Tenure distribution")
    axes[0].set_xlabel("Months")
    axes[0].legend()

    # Monthly charges comparison
    axes[1].hist(df_ref["MonthlyCharges"], bins=30, color="steelblue", alpha=0.6)
    axes[1].axvline(monthly_charges, color="tomato", linewidth=2, label="This customer")
    axes[1].set_title("Monthly charges distribution")
    axes[1].set_xlabel("$ per month")
    axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()