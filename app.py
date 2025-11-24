import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ===========================
# Load trained models
# ===========================
pipe_diab = joblib.load("diabetes_model.joblib")
pipe_ckd  = joblib.load("ckd_model.joblib")
pipe_cvd  = joblib.load("cvd_model.joblib")

st.set_page_config(page_title="Non-Dietary Chronic Disease Risk Tool", page_icon="🧬")

# ===========================
# Tabs
# ===========================
tab_public, tab_research = st.tabs(["📊 Public Risk Tool", "🔬 Researcher Tools"])

# ============================================================
# ============================================================
#               TAB 1 — PUBLIC RISK TOOL
# ============================================================
# ============================================================
with tab_public:

    st.title("Non-Dietary Chronic Disease Risk Assessment Tool")
    st.caption("Research prototype based on NHANES 2011–2020. Not for clinical use.")

    # Anthropometrics
    st.subheader("Anthropometrics")
    colA, colB = st.columns(2)

    with colA:
        height_unit = st.selectbox("Height unit", ["cm", "inches"])
        if height_unit == "cm":
            height_val = st.number_input("Height (cm)", 100.0, 250.0, 175.0)
            height_m = height_val / 100
        else:
            height_val = st.number_input("Height (inches)", 40.0, 100.0, 70.0)
            height_m = height_val * 0.0254

    with colB:
        weight_unit = st.selectbox("Weight unit", ["kg", "lbs"])
        if weight_unit == "kg":
            weight_val = st.number_input("Weight (kg)", 30.0, 300.0, 75.0)
            weight_kg = weight_val
        else:
            weight_val = st.number_input("Weight (lbs)", 60.0, 600.0, 165.0)
            weight_kg = weight_val * 0.453592

    bmi = weight_kg / (height_m ** 2)
    st.write(f"**Calculated BMI:** {bmi:.1f} kg/m²")

    # Income → poverty ratio
    st.subheader("Socioeconomic Variables")

    BASE_POVERTY_48 = {1:15650,2:21150,3:26650,4:32150,5:37650,6:43150,7:48650,8:54150}
    EXTRA_PER_PERSON_48 = 5500

    colF1, colF2 = st.columns(2)
    with colF1:
        family_income = st.number_input("Annual family income (USD)", 0, 300000, 60000)
    with colF2:
        household_size = st.selectbox("Household size", list(range(1, 13)), index=3)

    if household_size <= 8:
        poverty_threshold = BASE_POVERTY_48[household_size]
    else:
        poverty_threshold = BASE_POVERTY_48[8] + EXTRA_PER_PERSON_48 * (household_size - 8)

    st.write(f"**Estimated poverty threshold (48 states):** ${poverty_threshold:,.0f}")
    income_ratio = family_income / poverty_threshold if poverty_threshold else 0
    st.write(f"**Income-to-poverty ratio:** {income_ratio:.2f}")

    # Demographics & clinical
    st.subheader("Demographics & Clinical Measurements")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", 18, 90, 45)
        waist = st.number_input("Waist circumference (cm)", 50.0, 200.0, 95.0)
        sbp = st.number_input("Avg systolic BP (mmHg)", 80, 220, 120)
        smoker = st.selectbox("Current smoker?", ["No", "Yes"])

    with col2:
        dbp = st.number_input("Avg diastolic BP (mmHg)", 40, 140, 75)
        hr = st.number_input("Resting heart rate (bpm)", 40, 140, 70)
        activity = st.selectbox("Physical activity level", ["Low", "Moderate", "High"])

    education = st.selectbox(
        "Education level (NHANES categories)",
        [
            "Less than 9th grade",
            "9-11th grade (Includes 12th w/o diploma)",
            "High school graduate/GED or equivalent",
            "Some college or AA degree",
            "College graduate or above"
        ]
    )

    education_map = {
        "Less than 9th grade": 1,
        "9-11th grade (Includes 12th w/o diploma)": 2,
        "High school graduate/GED or equivalent": 3,
        "Some college or AA degree": 4,
        "College graduate or above": 5,
    }

    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race/ethnicity", ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"])

    gender_map = {"Male": 1, "Female": 2}
    race_map = {"Non-Hispanic White":1, "Non-Hispanic Black":2, "Hispanic":3, "Other":4}

    smoke_map = {"No": 0, "Yes": 1}
    activity_map = {"Low": 0, "Moderate": 1, "High": 2}

    if st.button("Estimate risks"):
        X = pd.DataFrame([{
            "bmi": bmi,
            "AgeYears": age,
            "waist_circumference": waist,
            "activity_level": activity_map[activity],
            "smoking": smoke_map[smoker],
            "avg_systolic": sbp,
            "avg_diastolic": dbp,
            "avg_HR": hr,
            "FamIncome_to_poverty_ratio": income_ratio,
            "Education": education_map[education],
            "Race": race_map[race],
            "Gender": gender_map[gender],
        }])

        p_diab = pipe_diab.predict_proba(X)[0, 1]
        p_ckd  = pipe_ckd.predict_proba(X)[0, 1]
        p_cvd  = pipe_cvd.predict_proba(X)[0, 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Diabetes Risk", f"{p_diab*100:.1f}%")
        c2.metric("CKD Risk", f"{p_ckd*100:.1f}%")
        c3.metric("CVD Risk", f"{p_cvd*100:.1f}%")

        st.caption("These estimates are based solely on non-dietary predictors.")


# ============================================================
# ============================================================
#               TAB 2 — RESEARCHER TOOLS
# ============================================================
# ============================================================
with tab_research:

    st.title("Researcher Tools: Variable Sensitivity Exploration")
    st.markdown("Systematically vary a predictor and observe resulting risk curves.")

    # -----------------------------
    # Choose variable to vary
    # -----------------------------
    variable_to_vary = st.selectbox(
        "Select a variable to vary:",
        [
            "AgeYears",
            "bmi",
            "waist_circumference",
            "avg_systolic",
            "avg_diastolic",
            "avg_HR",
            "FamIncome_to_poverty_ratio"
        ]
    )

    # Range
    min_val = st.number_input("Minimum value", value=20.0)
    max_val = st.number_input("Maximum value", value=80.0)
    n_points = st.slider("Number of points", 20, 200, 80)

    values = np.linspace(min_val, max_val, n_points)

    # --------------------------------------
    # Fixed defaults for all other variables
    # --------------------------------------
    st.subheader("Fixed values for all other predictors")

    colL, colR = st.columns(2)
    with colL:
        fix_age = st.number_input("Age (years)", 18, 90, 45)
        fix_bmi = st.number_input("BMI", 15.0, 60.0, 28.0)
        fix_waist = st.number_input("Waist circumference (cm)", 50.0, 200.0, 95.0)
        fix_income = st.number_input("Income-to-poverty ratio", 0.0, 10.0, 2.0)

    with colR:
        fix_sbp = st.number_input("Systolic BP", 80, 220, 120)
        fix_dbp = st.number_input("Diastolic BP", 40, 140, 75)
        fix_hr  = st.number_input("Heart rate", 40, 140, 70)

    # NHANES factor controls (not varied)
    education_fixed = 4
    race_fixed = 1
    gender_fixed = 1
    smoking_fixed = 0
    activity_fixed = 1

    if st.button("Generate sensitivity curves"):

        risks_diab = []
        risks_ckd = []
        risks_cvd = []

        for v in values:
            X = pd.DataFrame([{
                "bmi": fix_bmi if variable_to_vary!="bmi" else v,
                "AgeYears": fix_age if variable_to_vary!="AgeYears" else v,
                "waist_circumference": fix_waist if variable_to_vary!="waist_circumference" else v,
                "activity_level": activity_fixed,
                "smoking": smoking_fixed,
                "avg_systolic": fix_sbp if variable_to_vary!="avg_systolic" else v,
                "avg_diastolic": fix_dbp if variable_to_vary!="avg_diastolic" else v,
                "avg_HR": fix_hr if variable_to_vary!="avg_HR" else v,
                "FamIncome_to_poverty_ratio": fix_income if variable_to_vary!="FamIncome_to_poverty_ratio" else v,
                "Education": education_fixed,
                "Race": race_fixed,
                "Gender": gender_fixed,
            }])

            risks_diab.append(pipe_diab.predict_proba(X)[0,1])
            risks_ckd.append(pipe_ckd.predict_proba(X)[0,1])
            risks_cvd.append(pipe_cvd.predict_proba(X)[0,1])

        # Plot
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(values, risks_diab, label="Diabetes", linewidth=2)
        ax.plot(values, risks_ckd, label="CKD", linewidth=2)
        ax.plot(values, risks_cvd, label="CVD", linewidth=2)
        ax.set_xlabel(variable_to_vary)
        ax.set_ylabel("Predicted risk")
        ax.legend()
        ax.grid(alpha=0.3)

        st.pyplot(fig)

    st.markdown("---")
    st.markdown("This tool enables researchers to understand marginal effects of individual predictors on predicted risk across diseases.")
