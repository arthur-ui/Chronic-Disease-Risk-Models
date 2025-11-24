import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

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
#                    SHARED INPUTS (TOP)
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

# Federal poverty guidelines for 48 contiguous states & D.C.
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

# Compute poverty threshold automatically (48 contiguous states)
if household_size <= 8:
    poverty_threshold = BASE_POVERTY_48[household_size]
else:
    poverty_threshold = BASE_POVERTY_48[8] + EXTRA_PER_PERSON_48 * (household_size - 8)

st.write(
    f"**Estimated poverty threshold (48 states, {household_size} people):** "
    f"${poverty_threshold:,.0f}"
)

income_ratio = family_income / poverty_threshold if poverty_threshold > 0 else 0
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

# ============================================================
#                      EDUCATION (NHANES)
# ============================================================
education = st.selectbox(
    "Education level (NHANES categories)",
    [
        "Less than 9th grade",
        "9-11th grade (Includes 12th w/o diploma)",
        "High school graduate/GED or equivalent",
        "Some college or AA degree",
        "College graduate or above",
    ],
)

education_map = {
    "Less than 9th grade": 1,
    "9-11th grade (Includes 12th w/o diploma)": 2,
    "High school graduate/GED or equivalent": 3,
    "Some college or AA degree": 4,
    "College graduate or above": 5,
}

# ============================================================
#                      RACE & GENDER (NHANES)
# ============================================================
gender = st.selectbox("Gender", ["Male", "Female"])
race = st.selectbox(
    "Race/ethnicity",
    ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"],
)

gender_map = {"Male": 1, "Female": 2}
race_map = {
    "Non-Hispanic White": 1,
    "Non-Hispanic Black": 2,
    "Hispanic": 3,
    "Other": 4,
}

smoke_map = {"No": 0, "Yes": 1}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}

# ============================================================
#              BUILD BASE FEATURE ROW (SHARED)
# ============================================================
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
#                    TAB 1: INDIVIDUAL TOOL
# ============================================================
with tab1:
    st.subheader("Individual Multi-Disease Risk Estimate")

    if st.button("Estimate risks"):
        p_diab = float(pipe_diab.predict_proba(X_base)[0, 1])
        p_ckd  = float(pipe_ckd.predict_proba(X_base)[0, 1])
        p_cvd  = float(pipe_cvd.predict_proba(X_base)[0, 1])

        r1, r2, r3 = st.columns(3)
        r1.metric("Diabetes Risk", f"{p_diab*100:.1f}%")
        r2.metric("CKD Risk", f"{p_ckd*100:.1f}%")
        r3.metric("CVD Risk", f"{p_cvd*100:.1f}%")

        st.caption(
            "These estimates are based solely on non-dietary predictors "
            "(anthropometrics, vital signs, sociodemographics)."
        )

# ============================================================
#                 TAB 2: RESEARCHER TOOLS
# ============================================================
with tab2:
    st.subheader("Researcher tools: sensitivity analysis and risk surfaces")
    st.markdown(
        "Use these tools to explore how risk changes as you vary model inputs, "
        "holding other factors constant at the values above."
    )

    # --- helper: choose model by disease name ---
    def get_model(disease: str):
        if disease == "Diabetes":
            return pipe_diab
        elif disease == "CKD":
            return pipe_ckd
        elif disease == "CVD":
            return pipe_cvd
        else:
            raise ValueError("Unknown disease")

    # -------------------------------------------
    #   2.1 Univariate sensitivity curves
    # -------------------------------------------
    st.markdown("### Univariate sensitivity (one variable at a time)")

    colu1, colu2 = st.columns(2)
    with colu1:
        disease_uni = st.selectbox(
            "Disease (for univariate curve)",
            ["Diabetes", "CKD", "CVD"],
            index=0,
        )
    with colu2:
        feature_uni = st.selectbox(
            "Feature to vary",
            [
                "AgeYears",
                "bmi",
                "waist_circumference",
                "avg_systolic",
                "avg_diastolic",
                "avg_HR",
                "FamIncome_to_poverty_ratio",
            ],
        )

    # sensible ranges for features
    feature_ranges = {
        "AgeYears": (18, 90),
        "bmi": (15, 50),
        "waist_circumference": (60, 150),
        "avg_systolic": (90, 180),
        "avg_diastolic": (50, 110),
        "avg_HR": (40, 120),
        "FamIncome_to_poverty_ratio": (0.2, 5.0),
    }

    f_min, f_max = feature_ranges[feature_uni]
    n_points = st.slider("Number of points in curve", 30, 200, 80)

    if st.button("Plot univariate sensitivity"):
        model = get_model(disease_uni)

        values = np.linspace(f_min, f_max, n_points)
        X_grid = pd.concat([X_base] * n_points, ignore_index=True)
        X_grid[feature_uni] = values

        probs = model.predict_proba(X_grid)[:, 1]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(values, probs, linewidth=2.5)
        ax.set_xlabel(feature_uni, fontsize=12)
        ax.set_ylabel(f"Predicted risk of {disease_uni}", fontsize=12)
        ax.set_title(
            f"{disease_uni} risk vs {feature_uni} "
            "(all other predictors held constant)",
            fontsize=12,
        )
        ax.grid(True, linestyle=":", alpha=0.4)
        st.pyplot(fig)

    st.markdown("---")

    # -------------------------------------------
    #   2.2 Bivariate heatmaps (2D risk surfaces)
    # -------------------------------------------
    st.markdown("### Bivariate risk surfaces (heatmaps)")

    colh1, colh2, colh3 = st.columns(3)
    with colh1:
        disease_heat = st.selectbox(
            "Disease (for heatmap)",
            ["Diabetes", "CKD", "CVD"],
            index=0,
        )
    with colh2:
        x_feature = st.selectbox(
            "X-axis feature",
            ["AgeYears", "bmi", "avg_systolic", "waist_circumference"],
            index=0,
        )
    with colh3:
        y_feature = st.selectbox(
            "Y-axis feature",
            ["bmi", "avg_systolic", "avg_diastolic", "FamIncome_to_poverty_ratio"],
            index=1,
        )

    if x_feature == y_feature:
        st.warning("Please choose two different features for X and Y.")
    else:
        x_min, x_max = feature_ranges.get(x_feature, (0, 1))
        y_min, y_max = feature_ranges.get(y_feature, (0, 1))

        nx = st.slider("Number of X points", 20, 100, 40, key="nx")
        ny = st.slider("Number of Y points", 20, 100, 40, key="ny")

        if st.button("Plot 2D risk surface"):
            model = get_model(disease_heat)

            x_vals = np.linspace(x_min, x_max, nx)
            y_vals = np.linspace(y_min, y_max, ny)

            # build full grid at once
            grid_rows = []
            for xv in x_vals:
                for yv in y_vals:
                    row = base_row.copy()
                    row[x_feature] = xv
                    row[y_feature] = yv
                    grid_rows.append(row)

            X_grid2 = pd.DataFrame(grid_rows)
            probs2 = model.predict_proba(X_grid2)[:, 1]
            Z = probs2.reshape(nx, ny)

            fig2, ax2 = plt.subplots(figsize=(6, 5))
            im = ax2.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[y_min, y_max, x_min, x_max],
            )
            cbar = fig2.colorbar(im, ax=ax2)
            cbar.set_label(f"Predicted {disease_heat} risk", fontsize=11)

            ax2.set_xlabel(y_feature, fontsize=12)
            ax2.set_ylabel(x_feature, fontsize=12)
            ax2.set_title(
                f"{disease_heat} risk surface: {x_feature} vs {y_feature}", fontsize=12
            )
            st.pyplot(fig2)

# ===========================
# Footer
# ===========================
st.markdown("---")
st.markdown("**Model info**")
st.markdown("- Bagged decision trees with preprocessing pipeline")
st.markdown("- Inputs mirror NHANES preprocessing pipeline")
