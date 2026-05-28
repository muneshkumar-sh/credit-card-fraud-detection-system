import streamlit as st
import pandas as pd
import joblib
import os

# ---------- Paths ----------
BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "..", "models", "fraud_model.pkl")
data_path = os.path.join(BASE_DIR, "..", "dataset", "cleaned_creditcard.csv")

# ---------- Load Files ----------
model = joblib.load(model_path)
df = pd.read_csv(data_path)

# ---------- Page Config ----------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------- Session State ----------
if "sample_data" not in st.session_state:
    st.session_state.sample_data = df.iloc[0]

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "None"

# ---------- Title ----------
st.title("💳 Credit Card Fraud Detection System")
st.markdown("Professional Machine Learning Fraud Detection Dashboard")

# ---------- Dashboard Metrics ----------
col1, col2, col3, col4 = st.columns(4)

total = len(df)
fraud = int(df["Class"].sum())
legit = total - fraud
fraud_percent = (fraud / total) * 100

col1.metric("Total Transactions", f"{total:,}")
col2.metric("Legitimate Cases", f"{legit:,}")
col3.metric("Fraud Cases", fraud)
col4.metric("Fraud %", f"{fraud_percent:.2f}%")

st.markdown("---")

# ---------- Sidebar ----------
st.sidebar.header("📌 Transaction Input Options")

mode = st.sidebar.radio(
    "Choose Input Mode",
    ["Sample Data", "Select Dataset Row"]
)

# ---------- Sample Data Mode ----------
if mode == "Sample Data":

    st.sidebar.subheader("Quick Samples")

    # Visual Selection Indicator
    if st.session_state.selected_sample == "Fraud":
        st.sidebar.error("⚠️ Fraud Sample Selected")

    elif st.session_state.selected_sample == "Legit":
        st.sidebar.success("✅ Legitimate Sample Selected")

    colA, colB = st.sidebar.columns(2)

    if colA.button("⚠️ Fraud"):
        st.session_state.sample_data = df[df["Class"] == 1].sample(1).iloc[0]
        st.session_state.selected_sample = "Fraud"
        st.session_state.prediction_done = False
        st.rerun()

    if colB.button("✅ Legit"):
        st.session_state.sample_data = df[df["Class"] == 0].sample(1).iloc[0]
        st.session_state.selected_sample = "Legit"
        st.session_state.prediction_done = False
        st.rerun()

# ---------- Select Dataset Row Mode ----------
else:

    sample_type = st.sidebar.selectbox(
        "Transaction Type",
        ["Fraud", "Legitimate"]
    )

    if sample_type == "Fraud":
        filtered_df = df[df["Class"] == 1].reset_index(drop=True)
    else:
        filtered_df = df[df["Class"] == 0].reset_index(drop=True)

    row_index = st.sidebar.number_input(
        "Select Row Index",
        min_value=0,
        max_value=len(filtered_df)-1,
        value=0,
        step=1
    )

    st.session_state.sample_data = filtered_df.iloc[row_index]
    st.session_state.prediction_done = False

# ---------- Current Sample ----------
sample = st.session_state.sample_data

st.sidebar.markdown("---")
st.sidebar.subheader("✍️ Transaction Features")

# ---------- Inputs ----------
time = st.sidebar.number_input(
    "Time",
    min_value=0.0,
    value=max(0.0, float(sample["Time"]))
)

amount = st.sidebar.number_input(
    "Amount",
    min_value=0.0,
    value=max(0.0, float(sample["Amount"]))
)

features = []

for i in range(1, 29):
    col = f"V{i}"

    val = st.sidebar.number_input(
        col,
        value=float(sample[col]),
        format="%.6f"
    )

    features.append(val)

# ---------- Prediction Panel ----------
st.subheader("🔍 Prediction Panel")

if st.button("Predict Transaction", use_container_width=True):

    data = [time] + features + [amount]

    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    input_df = pd.DataFrame([data], columns=columns)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    st.session_state.prediction_done = True
    st.session_state.prediction = prediction
    st.session_state.probability = probability

# ---------- Prediction Result ----------
if st.session_state.prediction_done:

    prediction = st.session_state.prediction
    probability = st.session_state.probability

    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected")
        st.write(f"Fraud Confidence: {probability[1]*100:.2f}%")

    else:
        st.success("✅ Legitimate Transaction")
        st.write(f"Legitimate Confidence: {probability[0]*100:.2f}%")

# ---------- Current Transaction ----------
st.markdown("---")
st.subheader("📄 Current Transaction Data")
st.dataframe(sample.to_frame().T, use_container_width=True)

# ---------- Dataset Preview ----------
st.markdown("---")
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)