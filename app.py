import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Market Predictor", layout="centered")

st.title("Market Prediction System")

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "xgbost_model.joblib"
model = joblib.load(MODEL_PATH)

feature_cols = [
    'PCR', 'PRICE', 'price_rolling_mean',
    'total_open_int_CE', 'total_open_int_PE',
    'total_contracts_CE', 'total_contracts_PE',
    'total_oi', 'total_volume', 'chg_oi_total', 'oi_diff'
]

# ---------------- FUNCTION ----------------
def predict(sample_input):
    if isinstance(sample_input, dict):
        sample_input = pd.DataFrame([sample_input])

    input_data = sample_input[feature_cols]

    pred_price = model.predict(input_data)

    pred_direction = (pred_price > input_data['PRICE']).astype(int)

    result = input_data.copy()
    result['Predicted Price'] = pred_price
    result['Direction'] = pred_direction

    # Labels
    result['Market'] = result['Direction'].apply(lambda x: "Bullish " if x == 1 else "Bearish ")
    result['Signal'] = result['Direction'].apply(lambda x: "BUY " if x == 1 else "SELL ")

    return result


# ---------------- MODE ----------------
mode = st.radio("Select Mode", ["Manual Input", "CSV Input"])

# ================= MANUAL =================
if mode == "Manual Input":

    st.subheader(" Enter Feature Values")

    user_input = {}
    for col in feature_cols:
        user_input[col] = st.number_input(col, value=0.0)

    if st.button("Predict"):
        result = predict(user_input)

        st.subheader(" Result")
        st.dataframe(result)

        row = result.iloc[0]

        st.success(f"Predicted Price: {row['Predicted Price']:.2f}")

        if row['Direction'] == 1:
            st.success(" Bullish → BUY")
        else:
            st.error(" Bearish → SELL")


# ================= CSV =================
else:

    st.subheader("Upload CSV")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("### Input Data")
        st.dataframe(df.head())

        if st.button("Run Prediction"):
            result = predict(df)

            st.subheader("Results")
            st.dataframe(result)

            # Summary
            bullish = (result['Direction'] == 1).sum()
            bearish = (result['Direction'] == 0).sum()

            st.write(f"Bullish: {bullish}")
            st.write(f"Bearish: {bearish}")