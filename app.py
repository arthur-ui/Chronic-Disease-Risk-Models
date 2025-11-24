import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ===========================
# Load trained models (cached)
# ===========================
@st.cache_resource
def load_models():
    pipe_diab = joblib.load("diabetes_model.joblib")
    pipe_ckd  = joblib.load("ckd_model.joblib")
    pipe_cvd  = joblib.load("cvd_model.joblib")
    return pipe_diab, pipe_ckd, pipe_cvd

pipe_diab, pipe_ckd, pipe_cvd = load_models()

st.set_page_config(page_title="Non-Dietary Chronic Disease Risk Tool", page_icon="🧬")

st.title("Non-Dietary Chronic Disease Risk Assessment Tool")
st.caption("Research prototype based on NHANES 2011–2020. Not for clinical use.")

# ============================================================
#               HEIGHT + WEIGHT INPUT (with unit choice)
# ============================================================
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

# ============================================================
#      FAMILY INCOME → INCOME-TO-POVERTY RATIO (FIPR)
# ============================================================
st.subheader("Socioeconomic Variables")

BASE_POVERTY_48 = {
    1: 15650,
    2: 21150,
    3: 26650,
    4: 32150,
    5: 37650,
    6: 43150,
    7: 48650,
    8: 54150,
}
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

st.write(f"**Estimated poverty threshold:** ${poverty_threshold:,.0f}")

income_ratio = family_income / poverty_threshold if poverty_threshold else 0
st.write(f"**Income-to-poverty ratio:** {income_ratio:.2f}")

# ============================================================
#                 DEMOGRAPHICS & CLINICAL MEASUREMENTS
# ============================================================
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

# Education
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
race = st.selectbox(
    "Race/ethnicity",
    ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"]
)

gender_map = {"Male": 1, "Female": 2}
race_map = {
    "Non-Hispanic White": 1,
    "Non-Hispanic Black": 2,
    "Hispanic": 3,
    "Other": 4
}

smoke_map = {"No": 0, "Yes": 1}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}

base_row = {
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
}
X_base = pd.DataFrame([base_row])

# ============================================================
#                        TABS
# ============================================================
tab1, tab2 = st.tabs(["Individual risk tool", "Researcher tools"])

# ============================================================
#                    TAB 1
# ============================================================
with tab1:
    st.subheader("Individual Multi-Disease Risk Estimate")

    if st.button("Estimate risks"):
        p_diab = pipe_diab.predict_proba(X_base)[0, 1]
        p_ckd  = pipe_ckd.predict_proba(X_base)[0, 1]
        p_cvd  = pipe_cvd.predict_proba(X_base)[0, 1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Diabetes Risk", f"{p_diab*100:.1f}%")
        c2.metric("CKD Risk", f"{p_ckd*100:.1f}%")
        c3.metric("CVD Risk", f"{p_cvd*100:.1f}%")

# ============================================================
#                    TAB 2
# ============================================================
with tab2:

    st.subheader("Univariate Sensitivity Analysis")

    disease_uni = st.selectbox("Disease", ["Diabetes", "CKD", "CVD"])
    feature_uni = st.selectbox(
        "Feature to vary",
        ["AgeYears","bmi","waist_circumference","avg_systolic",
         "avg_diastolic","avg_HR","FamIncome_to_poverty_ratio"]
    )

    ranges = {
        "AgeYears": (18, 90),
        "bmi": (15, 50),
        "waist_circumference": (60, 150),
        "avg_systolic": (90, 180),
        "avg_diastolic": (50, 110),
        "avg_HR": (40, 120),
        "FamIncome_to_poverty_ratio": (0.2, 5),
    }
    fmin, fmax = ranges[feature_uni]

    n_points = st.slider("Resolution", 30, 200, 100)

    if st.button("Plot sensitivity"):
        vals = np.linspace(fmin, fmax, n_points)
        Xgrid = pd.concat([X_base]*n_points, ignore_index=True)
        Xgrid[feature_uni] = vals

        model = {"Diabetes":pipe_diab,"CKD":pipe_ckd,"CVD":pipe_cvd}[disease_uni]
        probs = model.predict_proba(Xgrid)[:,1]

        fig = px.line(x=vals, y=probs,
                      labels={"x":feature_uni, "y":f"{disease_uni} risk"},
                      title=f"Risk vs {feature_uni}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("2D Risk Surface (Heatmap)")

    disease_heat = st.selectbox("Disease (heatmap)", ["Diabetes","CKD","CVD"])
    x_feature = st.selectbox("X-axis feature", list(ranges.keys()), index=0)
    y_feature = st.selectbox("Y-axis feature", list(ranges.keys()), index=1)

    if x_feature == y_feature:
        st.warning("Choose two different features.")
    else:
        nx = st.slider("X resolution", 20, 100, 40)
        ny = st.slider("Y resolution", 20, 100, 40)

        if st.button("Plot heatmap"):
            xv = np.linspace(*ranges[x_feature], nx)
            yv = np.linspace(*ranges[y_feature], ny)

            grid = []
            for a in xv:
                for b in yv:
                    row = base_row.copy()
                    row[x_feature] = a
                    row[y_feature] = b
                    grid.append(row)

            X2 = pd.DataFrame(grid)
            model = {"Diabetes":pipe_diab,"CKD":pipe_ckd,"CVD":pipe_cvd}[disease_heat]
            Z = model.predict_proba(X2)[:,1].reshape(nx, ny)

            fig2 = go.Figure(data=go.Heatmap(
                z=Z,
                x=yv,
                y=xv,
                colorscale="Viridis"
            ))
            fig2.update_layout(
                title=f"{disease_heat} Risk Surface",
                xaxis_title=y_feature,
                yaxis_title=x_feature
            )
            st.plotly_chart(fig2, use_container_width=True)
