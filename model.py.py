import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="centered"
)

st.markdown("""
<style>

/* Dark overlay background */
.stApp {
    background-image: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
    url("https://images.unsplash.com/photo-1550751827-4bd374c3f58b");
    background-size: cover;
    background-position: center;
}

/* Main container */
.main-box {
    background-color: rgba(0,0,0,0.85);
    padding: 40px;
    border-radius: 15px;
}

/* Title */
.title {
    font-size: 48px;
    font-weight: bold;
    color: white;
    text-align: center;
    text-shadow: 2px 2px 10px black;
}

/* Subtitle */
.subtitle {
    font-size: 20px;
    color: white;
    text-align: center;
    margin-bottom: 30px;
}

/* Labels */
label {
    color: white !important;
    font-size: 16px !important;
}

/* Button */
.stButton>button {
    background-color: #00FFD1;
    color: black;
    font-size: 18px;
    border-radius: 10px;
    height: 50px;
    width: 100%;
}

.stButton>button:hover {
    background-color: #00c9a7;
    color: white;
}

.result-success {
    background-color: #00C853;
    color: white;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
    border-radius: 12px;
    margin-top: 25px;
    box-shadow: 0px 0px 25px #00C853;
}

.result-fraud {
    background-color: #D50000;
    color: white;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
    border-radius: 12px;
    margin-top: 25px;
    box-shadow: 0px 0px 25px #FF1744;
}
</style>
""", unsafe_allow_html=True)

model=joblib.load('Fraud Detection Dataset (3).pkl')
le1=joblib.load('le1.pkl')
le2=joblib.load('le2.pkl')
le3=joblib.load('le3.pkl')
le4=joblib.load('le4.pkl')
s=joblib.load('scalerfraud.pkl')

st.markdown('<div class="main-box">', unsafe_allow_html=True)

st.markdown('<div class="title">🛡️ Fraud Detection System</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle">Enter transaction details</div>', unsafe_allow_html=True)

amount = st.number_input("Transaction Amount")
transaction_type = st.selectbox("Transaction Type",le1.classes_)
time = st.number_input("Time of Transaction")
device = st.selectbox("Device Used",le2.classes_)
location = st.selectbox("Location",le3.classes_)
previous_fraud = st.number_input("Previous Fraudulent Transactions")
account_age = st.number_input("Account Age(days)")
transactions_24h = st.number_input("Number of Transactions Last 24 Hours",min_value=0)
payment_method = st.selectbox("Payment Method",le4.classes_)

transaction_type = le1.transform([transaction_type])[0]
device = le2.transform([device])[0]
location = le3.transform([location])[0]
payment_method = le4.transform([payment_method])[0]

if st.button("Predict"):

    input_data = np.array([[
        amount,
        transaction_type,
        time,
        device,
        location,
        previous_fraud,
        account_age,
        transactions_24h,
        payment_method
    ]])

    input_scaled = s.transform(input_data)

    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.markdown(
            '<div class="result-fraud">⚠ FRAUDULENT TRANSACTION DETECTED</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result-success">✅ LEGITIMATE TRANSACTION</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)