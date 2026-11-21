
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Water Potability Predictor",
    page_icon="💧",
    layout="wide"
)

MODEL_PATH = Path("best_water_potability_model.pkl")

st.title("💧 Water Potability Prediction")
st.markdown(
    "Enter the water-quality measurements below and the trained best classification model "
    "will predict whether the water is potable."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Run the final cells of the Water Potability notebook first "
        "to create `best_water_potability_model.pkl`, then run Streamlit again."
    )
    st.stop()

model = joblib.load(MODEL_PATH)

st.sidebar.header("Model")
st.sidebar.success("Best tuned model loaded")
st.sidebar.info("The model includes the preprocessing pipeline used during training.")

st.subheader("Water Quality Features")

col1, col2, col3 = st.columns(3)

with col1:
    ph = st.number_input(
        "pH",
        min_value=0.0,
        max_value=14.0,
        value=7.0,
        step=0.01
    )
    hardness = st.number_input(
        "Hardness",
        min_value=0.0,
        value=200.0,
        step=0.1
    )
    solids = st.number_input(
        "Solids",
        min_value=0.0,
        value=20000.0,
        step=100.0
    )

with col2:
    chloramines = st.number_input(
        "Chloramines",
        min_value=0.0,
        value=7.0,
        step=0.01
    )
    sulfate = st.number_input(
        "Sulfate",
        min_value=0.0,
        value=330.0,
        step=0.1
    )
    conductivity = st.number_input(
        "Conductivity",
        min_value=0.0,
        value=425.0,
        step=0.1
    )

with col3:
    organic_carbon = st.number_input(
        "Organic Carbon",
        min_value=0.0,
        value=14.0,
        step=0.01
    )
    trihalomethanes = st.number_input(
        "Trihalomethanes",
        min_value=0.0,
        value=66.0,
        step=0.1
    )
    turbidity = st.number_input(
        "Turbidity",
        min_value=0.0,
        value=4.0,
        step=0.01
    )

input_data = pd.DataFrame([{
    "ph": ph,
    "Hardness": hardness,
    "Solids": solids,
    "Chloramines": chloramines,
    "Sulfate": sulfate,
    "Conductivity": conductivity,
    "Organic_carbon": organic_carbon,
    "Trihalomethanes": trihalomethanes,
    "Turbidity": turbidity
}])

st.subheader("Input Data")
st.dataframe(input_data, use_container_width=True)

if st.button("🔍 Predict Potability", type="primary", use_container_width=True):
    prediction = int(model.predict(input_data)[0])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        probability_not_potable = probabilities[0]
        probability_potable = probabilities[1]
    else:
        probability_not_potable = None
        probability_potable = None

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("✅ The water is predicted to be POTABLE.")
    else:
        st.error("❌ The water is predicted to be NOT POTABLE.")

    if probability_potable is not None:
        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Potable Probability",
                f"{probability_potable * 100:.2f}%"
            )

        with c2:
            st.metric(
                "Not Potable Probability",
                f"{probability_not_potable * 100:.2f}%"
            )

        st.progress(float(probability_potable))

st.caption(
    "Note: This application is a machine-learning prediction demo and should not be "
    "used as a substitute for laboratory water-quality testing."
)
