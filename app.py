import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
pipe = joblib.load("diabetes_model.joblib")

st.set_page_config(page_title="Diabetes Risk Tool", page_icon="🩺")
st.title("Non-Dietary Diabetes Risk Assessment")
st.caption("Research prototype. Not for clinical use.")

# ---- User Inputs ----
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (years)", 18, 90, 45)
    bmi = st.number_input("BMI (kg/m²)", 10.0, 60.0, 27.0)
    waist = st.number_input("Waist circumference (cm)", 40.0, 200.0, 95.0)
    sbp = st.number_input("Average systolic BP (mmHg)", 80, 220, 120)

with col2:
    dbp = st.number_input("Average diastolic BP (mmHg)", 40, 140, 75)
    hr = st.number_input("Average resting heart rate (bpm)", 40, 150, 70)
    smoker = st.selectbox("Current smoker?", ["No", "Yes"])
    activity = st.selectbox("Physical activity level", ["Low", "Moderate", "High"])

income = st.number_input("Family income-to-poverty ratio", 0.0, 10.0, 2.0)

col3, col4 = st.columns(2)
with col3:
    gender = st.selectbox("Gender", ["Male", "Female"])
with col4:
    race = st.selectbox(
        "Race/ethnicity",
        ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"]
    )

# Map UI choices → NHANES codes
gender_map = {"Male": 1, "Female": 2}
race_map = {
    "Non-Hispanic White": 1,
    "Non-Hispanic Black": 2,
    "Hispanic": 3,
    "Other": 4,
}
smoke_map = {"No": 0, "Yes": 1}
activity_map = {"Low": 1, "Moderate": 2, "High": 3}

# ---- Prediction Button ----
if st.button("Estimate diabetes risk"):
    X_raw = pd.DataFrame([{
        "bmi": bmi,
        "AgeYears": age,
        "waist_circumference": waist,
        "activity_level": activity_map[activity],
        "smoking": smoke_map[smoker],
        "avg_systolic": sbp,
        "avg_diastolic": dbp,
        "avg_HR": hr,
        "FamIncome_to_poverty_ratio": income,
        "Education": None,  # Placeholder; can add later
        "Race": race_map[race],
        "Gender": gender_map[gender],
    }])

    proba = float(pipe.predict_proba(X_raw)[0, 1])
    st.metric("Estimated current diabetes risk", f"{proba*100:.1f}%")

    st.caption(
        "This estimate is based on non-dietary predictors only and is intended "
        "for research and educational purposes, not for diagnosis or treatment."
    )

# ---- Footer ----
st.markdown("---")
st.markdown("**Model details**")
st.markdown(
    "- Model: Bagged decision trees with preprocessing pipeline\n"
    "- Training data: NHANES 2011–2016, validated on 2017–2020\n"
    "- Inputs: anthropometrics, vital signs, sociodemographics"
)
