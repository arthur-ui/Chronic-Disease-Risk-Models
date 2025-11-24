import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

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

st.set_page_config(page_title="Non-Laboratory Chronic Disease Risk Tool", page_icon="🧬")

st.title("Non-Laboratory Chronic Disease Risk Assessment Tool")
st.caption("Research prototype based on NHANES 2011–2020. Not for clinical use.")

# ===========================
# Shared constants / mappings
# ===========================

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

education_map = {
    "Less than 9th grade": 1,
    "9-11th grade (Includes 12th w/o diploma)": 2,
    "High school graduate/GED or equivalent": 3,
    "Some college or AA degree": 4,
    "College graduate or above": 5,
}

gender_map = {"Male": 1, "Female": 2}
race_map = {
    "Non-Hispanic White": 1,
    "Non-Hispanic Black": 2,
    "Hispanic": 3,
    "Other": 4
}
smoke_map = {"No": 0, "Yes": 1}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}

# Default baseline profile (used if user hasn't run the risk calculator yet)
DEFAULT_BASELINE = {
    "bmi": 27.0,
    "AgeYears": 45,
    "waist_circumference": 95.0,
    "activity_level": 1,                 # Moderate
    "smoking": 0,                        # No
    "avg_systolic": 120,
    "avg_diastolic": 75,
    "avg_HR": 70,
    "FamIncome_to_poverty_ratio": 2.0,
    "Education": 4,                      # Some college/AA
    "Race": 1,                           # NH White
    "Gender": 1,                         # Male
}

# Friendly labels for researcher plots
VAR_LABELS = {
    "AgeYears": "Age (years)",
    "bmi": "Body mass index (kg/m²)",
    "waist_circumference": "Waist circumference (cm)",
    "avg_systolic": "Systolic BP (mmHg)",
    "avg_diastolic": "Diastolic BP (mmHg)",
    "avg_HR": "Heart rate (bpm)",
    "FamIncome_to_poverty_ratio": "Income-to-poverty ratio",
}

# Recommended ranges for sensitivity analysis
SENSITIVITY_RANGES = {
    "AgeYears": (20, 80),
    "bmi": (18, 40),
    "waist_circumference": (60, 130),
    "avg_systolic": (90, 180),
    "avg_diastolic": (50, 110),
    "avg_HR": (50, 110),
    "FamIncome_to_poverty_ratio": (0.5, 5.0),
}

# ===========================
# Tabs
# ===========================
tab_calc, tab_research = st.tabs(["Risk calculator", "Researcher tools"])

# ============================================================
#                      TAB 1: RISK CALCULATOR
# ============================================================
with tab_calc:
    st.subheader("Individual risk calculator")
    st.markdown(
        "Enter your information below. The model uses only non-laboratory predictors "
        "to estimate your 10-year risk of diabetes, chronic kidney disease (CKD), "
        "and cardiovascular disease (CVD)."
    )

    # ------------------ Anthropometrics ------------------
    st.markdown("### Anthropometrics")

    colA, colB = st.columns(2)

    with colA:
        height_unit = st.selectbox("Height unit", ["cm", "inches"])
        if height_unit == "cm":
            height_val = st.number_input("Height (cm)", 100.0, 250.0, 175.0)
            height_m = height_val / 100.0
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

    # ------------------ Socioeconomic variables ------------------
    st.markdown("### Socioeconomic variables")

    colF1, colF2 = st.columns(2)

    with colF1:
        family_income = st.number_input("Annual family income (USD)", 0, 300000, 60000)

    with colF2:
        household_size = st.selectbox("Household size", list(range(1, 13)), index=3)

    if household_size <= 8:
        poverty_threshold = BASE_POVERTY_48[household_size]
    else:
        poverty_threshold = BASE_POVERTY_48[8] + EXTRA_PER_PERSON_48 * (household_size - 8)

    st.write(
        f"**Estimated poverty threshold (48 contiguous states, {household_size} people):** "
        f"${poverty_threshold:,.0f}"
    )

    income_ratio = family_income / poverty_threshold if poverty_threshold > 0 else 0.0
    st.write(f"**Income-to-poverty ratio:** {income_ratio:.2f}")

    # ------------------ Demographics & clinical ------------------
    st.markdown("### Demographics & clinical measurements")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age (years)", 18, 90, 45)
        waist = st.number_input("Waist circumference (cm)", 50.0, 200.0, 95.0)
        sbp = st.number_input("Average systolic BP (mmHg)", 80, 220, 120)
        smoker = st.selectbox("Current smoker?", ["No", "Yes"])

    with col2:
        dbp = st.number_input("Average diastolic BP (mmHg)", 40, 140, 75)
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

    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox(
        "Race/ethnicity",
        ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"]
    )

    # ------------------ Build row & predict ------------------
    if st.button("Estimate risks", type="primary"):
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

        # Save baseline for researcher tools
        st.session_state["baseline_features"] = X.iloc[0].to_dict()

        p_diab = float(pipe_diab.predict_proba(X)[0, 1])
        p_ckd  = float(pipe_ckd.predict_proba(X)[0, 1])
        p_cvd  = float(pipe_cvd.predict_proba(X)[0, 1])

        r1, r2, r3 = st.columns(3)
        r1.metric("Diabetes risk", f"{p_diab * 100:.1f} %")
        r2.metric("CKD risk", f"{p_ckd * 100:.1f} %")
        r3.metric("CVD risk", f"{p_cvd * 100:.1f} %")

        st.caption(
            "These estimates are based solely on non-laboratory predictors and are "
            "intended for research and public health exploration, not for clinical decision-making."
        )

    st.markdown("---")
    st.markdown("**Model info**")
    st.markdown("- Bagged decision trees with preprocessing pipeline")
    st.markdown("- Inputs mirror NHANES preprocessing pipeline")


# ============================================================
#                      TAB 2: RESEARCHER TOOLS
# ============================================================
with tab_research:
    st.subheader("Researcher tools: sensitivity analysis and risk landscapes")

    st.markdown(
        "This section allows you to explore how the model's predicted risks change as you vary key predictors. "
        "By default, we use the last profile you entered on the *Risk calculator* tab; if none is available, "
        "we use a typical baseline profile."
    )

    # Get baseline features
    baseline = st.session_state.get("baseline_features", DEFAULT_BASELINE)

    # Show baseline profile
    with st.expander("Show baseline profile used in analyses", expanded=False):
        friendly = {
            "Age (years)": baseline["AgeYears"],
            "BMI (kg/m²)": baseline["bmi"],
            "Waist circumference (cm)": baseline["waist_circumference"],
            "Systolic BP (mmHg)": baseline["avg_systolic"],
            "Diastolic BP (mmHg)": baseline["avg_diastolic"],
            "Heart rate (bpm)": baseline["avg_HR"],
            "Income-to-poverty ratio": baseline["FamIncome_to_poverty_ratio"],
            "Smoking (0=No,1=Yes)": baseline["smoking"],
            "Activity level (0=Low,1=Moderate,2=High)": baseline["activity_level"],
            "Education (1–5)": baseline["Education"],
            "Race (1–4)": baseline["Race"],
            "Gender (1=Male,2=Female)": baseline["Gender"],
        }
        st.json(friendly)

    # Helper: build DataFrame from baseline dict
    def df_from_baseline(baseline_dict):
        return pd.DataFrame([baseline_dict])

    # ------------------ 1D sensitivity analysis ------------------
    st.markdown("### 1D sensitivity analysis (all three diseases)")

    st.markdown(
        "Select a variable to vary while holding the rest of the profile constant. "
        "The plot will show predicted risk for **diabetes, CKD, and CVD** as that variable changes."
    )

    var_options = list(SENSITIVITY_RANGES.keys())
    var_choice = st.selectbox(
        "Variable to vary",
        options=var_options,
        format_func=lambda v: VAR_LABELS.get(v, v)
    )

    vmin_default, vmax_default = SENSITIVITY_RANGES[var_choice]

    # Choose range and resolution
    col_range1, col_range2 = st.columns(2)
    with col_range1:
        vmin = st.number_input(
            f"Minimum {VAR_LABELS.get(var_choice, var_choice)}",
            value=float(vmin_default)
        )
    with col_range2:
        vmax = st.number_input(
            f"Maximum {VAR_LABELS.get(var_choice, var_choice)}",
            value=float(vmax_default)
        )

    step = st.number_input("Step size", value=2.0 if var_choice != "FamIncome_to_poverty_ratio" else 0.25, min_value=0.01)

    if vmin >= vmax:
        st.error("Minimum must be less than maximum.")
    else:
        if st.button("Run 1D sensitivity analysis", key="run_1d_sens"):
            x_vals = np.arange(vmin, vmax + 1e-9, step)

            results = []
            for val in x_vals:
                row = baseline.copy()
                row[var_choice] = float(val)
                X = df_from_baseline(row)

                p_diab = float(pipe_diab.predict_proba(X)[0, 1])
                p_ckd  = float(pipe_ckd.predict_proba(X)[0, 1])
                p_cvd  = float(pipe_cvd.predict_proba(X)[0, 1])

                results.append({"Variable": val, "Disease": "Diabetes", "Risk": p_diab})
                results.append({"Variable": val, "Disease": "CKD",      "Risk": p_ckd})
                results.append({"Variable": val, "Disease": "CVD",      "Risk": p_cvd})

            df_plot = pd.DataFrame(results)

            fig_line = px.line(
                df_plot,
                x="Variable",
                y="Risk",
                color="Disease",
                labels={
                    "Variable": VAR_LABELS.get(var_choice, var_choice),
                    "Risk": "Predicted risk (probability)",
                    "Disease": "Outcome"
                },
            )
            fig_line.update_layout(
                yaxis=dict(range=[0, 1]),
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # ------------------ 2D heatmap (moved here) ------------------
    st.markdown("### 2D risk heatmap")

    st.markdown(
        "Generate a risk heatmap by varying **two predictors** at once for a selected disease, "
        "holding all other characteristics at the baseline profile."
    )

    disease_choice = st.selectbox("Disease for heatmap", ["Diabetes", "CKD", "CVD"])

    # Subset of variables for heatmap (keep it manageable)
    heatmap_vars = ["AgeYears", "bmi", "waist_circumference", "avg_systolic", "avg_diastolic"]

    col_hx, col_hy = st.columns(2)
    with col_hx:
        x_var = st.selectbox(
            "X-axis variable",
            options=heatmap_vars,
            format_func=lambda v: VAR_LABELS.get(v, v),
            key="x_var_heat"
        )
    with col_hy:
        y_var = st.selectbox(
            "Y-axis variable",
            options=heatmap_vars,
            format_func=lambda v: VAR_LABELS.get(v, v),
            key="y_var_heat"
        )

    if x_var == y_var:
        st.warning("X and Y variables must be different to generate a heatmap.")
    else:
        # Ranges
        xmin, xmax = SENSITIVITY_RANGES[x_var]
        ymin, ymax = SENSITIVITY_RANGES[y_var]

        st.markdown("Specify ranges for the heatmap axes:")

        col_xr1, col_xr2 = st.columns(2)
        with col_xr1:
            x_min_val = st.number_input(
                f"X min ({VAR_LABELS.get(x_var, x_var)})",
                value=float(xmin),
                key="x_min_heat"
            )
        with col_xr2:
            x_max_val = st.number_input(
                f"X max ({VAR_LABELS.get(x_var, x_var)})",
                value=float(xmax),
                key="x_max_heat"
            )

        col_yr1, col_yr2 = st.columns(2)
        with col_yr1:
            y_min_val = st.number_input(
                f"Y min ({VAR_LABELS.get(y_var, y_var)})",
                value=float(ymin),
                key="y_min_heat"
            )
        with col_yr2:
            y_max_val = st.number_input(
                f"Y max ({VAR_LABELS.get(y_var, y_var)})",
                value=float(ymax),
                key="y_max_heat"
            )

        grid_res = st.slider("Grid resolution (higher = smoother, slower)", 20, 80, 40)

        if x_min_val >= x_max_val or y_min_val >= y_max_val:
            st.error("Minimums must be less than maximums for both axes.")
        else:
            if st.button("Generate heatmap", key="run_heatmap"):
                x_values = np.linspace(x_min_val, x_max_val, grid_res)
                y_values = np.linspace(y_min_val, y_max_val, grid_res)

                risk_grid = np.zeros((len(y_values), len(x_values)))

                if disease_choice == "Diabetes":
                    model = pipe_diab
                elif disease_choice == "CKD":
                    model = pipe_ckd
                else:
                    model = pipe_cvd

                for i, y_val in enumerate(y_values):
                    for j, x_val in enumerate(x_values):
                        row = baseline.copy()
                        row[x_var] = float(x_val)
                        row[y_var] = float(y_val)
                        X_grid = df_from_baseline(row)
                        risk_grid[i, j] = float(model.predict_proba(X_grid)[0, 1])

                fig_hm = px.imshow(
                    risk_grid,
                    x=np.round(x_values, 1),
                    y=np.round(y_values, 1),
                    labels=dict(
                        x=VAR_LABELS.get(x_var, x_var),
                        y=VAR_LABELS.get(y_var, y_var),
                        color="Predicted risk"
                    ),
                    origin="lower",
                    aspect="auto",
                    color_continuous_scale="Viridis",
                )
                fig_hm.update_coloraxes(cmin=0, cmax=1)
                st.plotly_chart(fig_hm, use_container_width=True)

    st.caption(
        "These researcher tools are intended to help visualize the model's behavior across the predictor space "
        "for both public health applications and methodological exploration."
    )
