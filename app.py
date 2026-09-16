from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", page_icon="🤖", layout="wide")
BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "best_model.joblib"
META_PATH = BASE / "models" / "metadata.json"
METRICS_PATH = BASE / "models" / "metrics.json"

st.title("🤖 Customer Churn Prediction — Machine Learning")
st.caption("Predict whether a customer is likely to churn from their account and service information.")

if not MODEL_PATH.exists() or not META_PATH.exists():
    st.warning("The trained model is not available yet.")
    st.code("python train.py", language="powershell")
    st.info("Run the command in the project folder. It will create the model files inside models/.")
    st.stop()

model = joblib.load(MODEL_PATH)
metadata = json.loads(META_PATH.read_text(encoding="utf-8"))
metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8")) if METRICS_PATH.exists() else {}

st.sidebar.header("Customer Information")

def select(label, options, default=None):
    return st.sidebar.selectbox(label, options, index=options.index(default) if default in options else 0)

gender = select("Gender", ["Female", "Male"], "Female")
senior = select("Senior Citizen", [0, 1], 0)
partner = select("Partner", ["No", "Yes"], "No")
dependents = select("Dependents", ["No", "Yes"], "No")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone = select("Phone Service", ["No", "Yes"], "Yes")
multiple = select("Multiple Lines", ["No phone service", "No", "Yes"], "No")
internet = select("Internet Service", ["DSL", "Fiber optic", "No"], "DSL")
online_security = select("Online Security", ["No internet service", "No", "Yes"], "No")
online_backup = select("Online Backup", ["No internet service", "No", "Yes"], "No")
device = select("Device Protection", ["No internet service", "No", "Yes"], "No")
tech = select("Tech Support", ["No internet service", "No", "Yes"], "No")
tv = select("Streaming TV", ["No internet service", "No", "Yes"], "No")
movies = select("Streaming Movies", ["No internet service", "No", "Yes"], "No")
contract = select("Contract", ["Month-to-month", "One year", "Two year"], "Month-to-month")
paperless = select("Paperless Billing", ["No", "Yes"], "Yes")
payment = select("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"], "Electronic check")
monthly = st.sidebar.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
total = st.sidebar.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=max(monthly * max(tenure, 1), 70.0), step=10.0)

row = pd.DataFrame([{
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
    "OnlineSecurity": online_security, "OnlineBackup": online_backup, "DeviceProtection": device,
    "TechSupport": tech, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
}])

if st.button("Predict Churn", type="primary", use_container_width=True):
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    label = "CHURN" if prediction == 1 else "NO CHURN"
    a, b = st.columns(2)
    a.metric("Prediction", label)
    b.metric("Churn Probability", f"{probability:.1%}")
    if prediction == 1:
        st.error("This customer profile has a higher predicted likelihood of churn.")
    else:
        st.success("This customer profile has a lower predicted likelihood of churn.")

st.divider()
st.subheader("Model Comparison")
rows = []
for name, m in metrics.items():
    rows.append({"Model": name, "Accuracy": m["accuracy"], "Precision": m["precision"], "Recall": m["recall"], "F1": m["f1"], "ROC-AUC": m["roc_auc"]})
if rows:
    results_df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    st.dataframe(results_df.style.format({"Accuracy":"{:.2%}","Precision":"{:.2%}","Recall":"{:.2%}","F1":"{:.2%}","ROC-AUC":"{:.2%}"}), use_container_width=True)
    fig = px.bar(results_df, x="Model", y="F1", title="F1 Score Comparison", text_auto=".2f")
    st.plotly_chart(fig, use_container_width=True)

st.info(f"Best model selected by F1 score: **{metadata['best_model']}**")
st.caption("The model estimates probability from patterns learned in the dataset; it is not a guarantee about an individual customer.")
