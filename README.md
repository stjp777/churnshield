# ChurnShield

A machine learning web application that predicts whether a telecom customer is likely to cancel their subscription. Built end-to-end in Python — from raw data to a live deployed app.

**Live demo:** https://churnshield777.streamlit.app

---

## What it does

Enter a customer's details (contract type, monthly charges, tenure, payment method, etc.) and ChurnShield will:

- Predict their churn probability
- Flag which risk factors are present
- Recommend specific retention actions
- Show where the customer sits relative to the full dataset

---

## Tech stack

| Layer | Tools |
|---|---|
| Data analysis | pandas, NumPy |
| Machine learning | scikit-learn |
| Visualization | matplotlib, seaborn |
| Web app | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Model

- **Algorithm:** Random Forest Classifier (100 estimators)
- **Dataset:** IBM Telco Customer Churn — 7,043 customer records, 21 features
- **Accuracy:** 79% on held-out test set
- **Class imbalance:** Handled with `class_weight="balanced"`
- **Train/test split:** 80/20

Top predictive features identified by feature importance:
1. Total charges
2. Monthly charges
3. Tenure
4. Contract type (two-year)
5. Internet service type (fiber optic)

---

## Run locally

```bash
# Clone the repo
git clone https://github.com/stjp777/churnshield.git
cd churnshield

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## Project structure

```
churnshield/
├── app.py               # Streamlit web application
├── churn_model.pkl      # Trained Random Forest model
├── model_columns.pkl    # Feature column names for input encoding
├── churn_clean.csv      # Cleaned, encoded dataset
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Dataset

IBM Telco Customer Churn dataset. Each record represents one customer with attributes including contract type, internet service, monthly charges, tenure, and whether they churned.

Source: [IBM Sample Data Sets](https://github.com/IBM/telco-customer-churn-on-icp4d)

---

## Author

Steve Nguyen — [github.com/stjp777](https://github.com/stjp777)
