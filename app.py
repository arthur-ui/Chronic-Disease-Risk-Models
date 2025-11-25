import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===========================
# Load trained models
# ===========================
pipe_diab = joblib.load("diabetes_model.joblib")
pipe_ckd  = joblib.load("ckd_model.joblib")
pipe_cvd  = joblib.load("cvd_model.joblib")

st.set_page_config(page_title="Non-Dietary Chronic Disease Risk Tool", page_icon="🧬")

# -------------------------------------------------
# Shared constants / helper functions
# -------------------------------------------------
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
    "Other": 4,
}
smoke_map = {"No": 0, "Yes": 1}
activity_map = {"Low": 0, "Moderate": 1, "High": 2}

# reverse maps for labelling subgroups
gender_rev = {v: k for k, v in gender_map.items()}
race_rev = {v: k for k, v in race_map.items()}
education_rev = {v: k for k, v in education_map.items()}
activity_rev = {0: "Low", 1: "Moderate", 2: "High"}
smoking_rev = {0: "No", 1: "Yes"}

def compute_poverty_threshold(household_size: int) -> int:
    if household_size <= 8:
        return BASE_POVERTY_48[household_size]
    return BASE_POVERTY_48[8] + EXTRA_PER_PERSON_48 * (household_size - 8)


def build_feature_df(
    bmi, age, waist, activity_label, smoker_label,
    sbp, dbp, hr, income_ratio, education_label,
    race_label, gender_label
) -> pd.DataFrame:
    return pd.DataFrame([{
        "bmi": bmi,
        "AgeYears": age,
        "waist_circumference": waist,
        "activity_level": activity_map[activity_label],
        "smoking": smoke_map[smoker_label],
        "avg_systolic": sbp,
        "avg_diastolic": dbp,
        "avg_HR": hr,
        "FamIncome_to_poverty_ratio": income_ratio,
        "Education": education_map[education_label],
        "Race": race_map[race_label],
        "Gender": gender_map[gender_label],
    }])


def predict_three(X: pd.DataFrame):
    """Return probabilities for diabetes, CKD, CVD as tuple of 1D numpy arrays."""
    p_diab = pipe_diab.predict_proba(X)[:, 1]
    p_ckd  = pipe_ckd.predict_proba(X)[:, 1]
    p_cvd  = pipe_cvd.predict_proba(X)[:, 1]
    return p_diab, p_ckd, p_cvd


# -------------------------------------------------
# Load NHANES test data for simulation
# -------------------------------------------------
# This file should live alongside the app script in the repo
nhanes_raw = pd.read_csv("nhanes_test_for_sim.csv")

# Features the models expect (same as baseline_df / base_df)
feature_cols_sim = [
    "bmi",
    "AgeYears",
    "waist_circumference",
    "activity_level",
    "smoking",
    "avg_systolic",
    "avg_diastolic",
    "avg_HR",
    "FamIncome_to_poverty_ratio",
    "Education",
    "Race",
    "Gender",
]

# Create a cleaned version for simulation: only needed columns,
# numeric coercion + mean imputation for missing values
nhanes_sim = nhanes_raw.copy()
for col in feature_cols_sim:
    if col in nhanes_sim.columns:
        nhanes_sim[col] = pd.to_numeric(nhanes_sim[col], errors="coerce")

# Impute missing values per column with column mean
for col in feature_cols_sim:
    if col in nhanes_sim.columns:
        col_mean = nhanes_sim[col].mean(skipna=True)
        nhanes_sim[col] = nhanes_sim[col].fillna(col_mean)

# Restrict to only the columns we actually use in the simulator
nhanes_sim = nhanes_sim[feature_cols_sim].copy()


# ===========================
# Tabs
# ===========================
tab_calc, tab_research = st.tabs(["Risk calculator", "Researcher tools"])


# ============================================================
#                     TAB 1: RISK CALCULATOR
# ============================================================
with tab_calc:
    st.title("Non-Dietary Chronic Disease Risk Assessment Tool")
    st.caption("Research prototype based on NHANES 2011–2020. Not for clinical use.")

    # ---------------- Anthropometrics ----------------
    st.subheader("Anthropometrics")

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

    # ---------------- Socioeconomic ----------------
    st.subheader("Socioeconomic variables")

    colF1, colF2 = st.columns(2)
    with colF1:
        family_income = st.number_input("Annual family income (USD)", 0, 300000, 60000)
    with colF2:
        household_size = st.selectbox("Household size", list(range(1, 13)), index=3)

    poverty_threshold = compute_poverty_threshold(household_size)
    st.write(
        f"**Estimated poverty threshold (48 states, {household_size} people):** "
        f"${poverty_threshold:,.0f}"
    )
    income_ratio = family_income / poverty_threshold if poverty_threshold > 0 else 0.0
    st.write(f"**Income-to-poverty ratio:** {income_ratio:.2f}")

    # ---------------- Demographics & clinical ----------------
    st.subheader("Demographics & clinical measurements")

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

    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox(
        "Race/ethnicity",
        ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"]
    )

    # ---------------- Predict ----------------
    if st.button("Estimate risks", key="calc_button"):
        X = build_feature_df(
            bmi=bmi, age=age, waist=waist, activity_label=activity,
            smoker_label=smoker, sbp=sbp, dbp=dbp, hr=hr,
            income_ratio=income_ratio, education_label=education,
            race_label=race, gender_label=gender
        )

        p_diab, p_ckd, p_cvd = predict_three(X)

        r1, r2, r3 = st.columns(3)
        r1.metric("Diabetes risk", f"{p_diab[0]*100:.1f}%")
        r2.metric("CKD risk", f"{p_ckd[0]*100:.1f}%")
        r3.metric("CVD risk", f"{p_cvd[0]*100:.1f}%")

        st.caption("These estimates are based solely on non-dietary predictors.")

    st.markdown("---")
    st.markdown("**Model info**")
    st.markdown("- Bagged decision trees with preprocessing pipeline")
    st.markdown("- Inputs mirror NHANES preprocessing pipeline")


# ============================================================
#                 TAB 2: RESEARCHER TOOLS
# ============================================================
with tab_research:
    st.title("Researcher tools & sensitivity analysis")
    st.caption(
        "Explore how changes in individual predictors affect modelled risk for "
        "diabetes, CKD, and CVD. All calculations use the same non-laboratory "
        "models as the main risk calculator."
    )

    # ---------------- Baseline profile for research tools ----------------
    st.subheader("Baseline profile (held constant for sensitivity analyses)")
    colB1, colB2, colB3 = st.columns(3)

    with colB1:
        age_r = st.number_input("Age (years)", 18, 90, 50, key="age_r")
        bmi_r = st.number_input("BMI (kg/m²)", 15.0, 60.0, 28.0, key="bmi_r")
        waist_r = st.number_input("Waist circumference (cm)", 50.0, 200.0, 100.0, key="waist_r")

    with colB2:
        sbp_r = st.number_input("Systolic BP (mmHg)", 80, 220, 135, key="sbp_r")
        dbp_r = st.number_input("Diastolic BP (mmHg)", 40, 140, 80, key="dbp_r")
        hr_r  = st.number_input("Resting heart rate (bpm)", 40, 140, 72, key="hr_r")

    with colB3:
        family_income_r = st.number_input("Annual family income (USD)", 0, 300000, 60000, key="inc_r")
        household_size_r = st.selectbox("Household size", list(range(1, 13)), index=3, key="hh_r")

    poverty_threshold_r = compute_poverty_threshold(household_size_r)
    income_ratio_r = family_income_r / poverty_threshold_r if poverty_threshold_r > 0 else 0.0
    st.write(f"Baseline income-to-poverty ratio: **{income_ratio_r:.2f}**")

    colB4, colB5, colB6 = st.columns(3)
    with colB4:
        education_r = st.selectbox(
            "Education (NHANES categories)", list(education_map.keys()),
            index=3, key="edu_r"
        )
    with colB5:
        gender_r = st.selectbox("Gender", ["Male", "Female"], key="gender_r")
    with colB6:
        race_r = st.selectbox(
            "Race/ethnicity",
            ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"],
            key="race_r"
        )

    baseline_df = build_feature_df(
        bmi=bmi_r, age=age_r, waist=waist_r,
        activity_label="Moderate",  # default; could expose later
        smoker_label="No",
        sbp=sbp_r, dbp=dbp_r, hr=hr_r,
        income_ratio=income_ratio_r,
        education_label=education_r,
        race_label=race_r, gender_label=gender_r
    )

    base_diab, base_ckd, base_cvd = predict_three(baseline_df)
    st.markdown(
        f"Baseline predicted risks – Diabetes: **{base_diab[0]*100:.2f}%**, "
        f"CKD: **{base_ckd[0]*100:.2f}%**, CVD: **{base_cvd[0]*100:.2f}%**."
    )

    st.markdown("---")

    # ============================================================
    # 1. One-dimensional sensitivity curves (all 3 diseases)
    # ============================================================
    st.subheader("One-dimensional sensitivity analysis")

    var_options = {
        "Age (years)": ("AgeYears", 20, 85, 40),
        "BMI (kg/m²)": ("bmi", 18, 45, 40),
        "Waist circumference (cm)": ("waist_circumference", 60, 140, 40),
        "Systolic BP (mmHg)": ("avg_systolic", 90, 180, 40),
        "Diastolic BP (mmHg)": ("avg_diastolic", 50, 110, 40),
        "Resting HR (bpm)": ("avg_HR", 50, 110, 40),
        "Income-to-poverty ratio": ("FamIncome_to_poverty_ratio", 0.3, 5.0, 40),
    }

    sens_label = st.selectbox(
        "Choose variable to vary",
        list(var_options.keys()),
        index=0
    )
    var_col, vmin, vmax, n_points = var_options[sens_label]
    vals = np.linspace(vmin, vmax, n_points)

    sens_X = pd.concat([baseline_df] * n_points, ignore_index=True)
    sens_X[var_col] = vals

    s_diab, s_ckd, s_cvd = predict_three(sens_X)

    sens_df = pd.DataFrame({
        "Value": vals,
        "Diabetes": s_diab,
        "CKD": s_ckd,
        "CVD": s_cvd,
    }).melt("Value", var_name="Disease", value_name="Risk")

    sens_chart = (
        alt.Chart(sens_df)
        .mark_line()
        .encode(
            x=alt.X("Value:Q", title=sens_label),
            y=alt.Y("Risk:Q", title="Predicted risk"),
            color=alt.Color("Disease:N", title=None),
        )
        .properties(height=300)
    )

    st.altair_chart(sens_chart, use_container_width=True)
    st.caption("Risk curves are generated by varying one predictor while holding the baseline profile constant.")

    st.markdown("---")

    # ============================================================
    # 2. Two-variable interaction heatmaps (Plotly, 3 panels)
    # ============================================================
    st.subheader("Two-variable interaction heatmaps")

    two_d_var_options = {
        "Age (years)": ("AgeYears", 20, 85),
        "BMI (kg/m²)": ("bmi", 18, 45),
        "Waist circumference (cm)": ("waist_circumference", 60, 140),
        "Systolic BP (mmHg)": ("avg_systolic", 90, 180),
        "Diastolic BP (mmHg)": ("avg_diastolic", 50, 110),
        "Resting HR (bpm)": ("avg_HR", 50, 110),
        "Income-to-poverty ratio": ("FamIncome_to_poverty_ratio", 0.3, 5.0),
    }

    colH1, colH2 = st.columns(2)
    with colH1:
        heat_x_label = st.selectbox(
            "X-axis variable",
            list(two_d_var_options.keys()),
            index=3  # default SBP
        )
    with colH2:
        y_choices = [k for k in two_d_var_options.keys() if k != heat_x_label]
        heat_y_label = st.selectbox(
            "Y-axis variable",
            y_choices,
            index=1
        )

    (x_col, x_min, x_max) = two_d_var_options[heat_x_label]
    (y_col, y_min, y_max) = two_d_var_options[heat_y_label]

    n_x = st.slider("Resolution (X)", 10, 60, 25, key="nx_heat")
    n_y = st.slider("Resolution (Y)", 10, 60, 25, key="ny_heat")

    x_vals = np.linspace(x_min, x_max, n_x)
    y_vals = np.linspace(y_min, y_max, n_y)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    n_grid = X_grid.size

    grid_X = pd.concat([baseline_df] * n_grid, ignore_index=True)
    grid_X[x_col] = X_grid.ravel()
    grid_X[y_col] = Y_grid.ravel()

    h_diab, h_ckd, h_cvd = predict_three(grid_X)

    z_diab = h_diab.reshape(n_y, n_x)
    z_ckd  = h_ckd.reshape(n_y, n_x)
    z_cvd  = h_cvd.reshape(n_y, n_x)

    max_risk = float(
        max(z_diab.max(), z_ckd.max(), z_cvd.max())
    )

    fig_heat = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Diabetes", "CKD", "CVD"),
        horizontal_spacing=0.06
    )

    for idx, (name, z) in enumerate(
        [("Diabetes", z_diab), ("CKD", z_ckd), ("CVD", z_cvd)],
        start=1
    ):
        fig_heat.add_trace(
            go.Heatmap(
                x=x_vals,
                y=y_vals,
                z=z,
                coloraxis="coloraxis"
            ),
            row=1, col=idx
        )

    fig_heat.update_layout(
        coloraxis=dict(colorscale="Viridis", cmin=0.0, cmax=max_risk),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    fig_heat.update_xaxes(title_text=heat_x_label, row=1, col=1)
    fig_heat.update_xaxes(title_text=heat_x_label, row=1, col=2)
    fig_heat.update_xaxes(title_text=heat_x_label, row=1, col=3)
    fig_heat.update_yaxes(title_text=heat_y_label, row=1, col=1)
    fig_heat.update_yaxes(title_text=heat_y_label, row=1, col=2)
    fig_heat.update_yaxes(title_text=heat_y_label, row=1, col=3)

    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        "Heatmaps show predicted risk across a 2D grid of values for the selected predictors, "
        "with all other variables fixed at the baseline profile."
    )

    st.markdown("---")

    # ============================================================
    # 3. Population scenario simulator (multi-variable changes)
    #      using jittered real NHANES rows
    # ============================================================
    st.subheader("Population scenario simulator (NHANES-based)")

    st.markdown(
        "Sample real NHANES participants, optionally jitter their values, then "
        "modify any combination of predictors and compare predicted risk before "
        "and after the scenario overall and across subgroups."
    )

    colP1, colP2, colP3 = st.columns(3)
    with colP1:
        pop_n = st.slider("Population size", 200, 20000, 3000, step=200, key="pop_n")
    with colP2:
        seed = st.number_input("Random seed", min_value=0, max_value=10000, value=42, key="seed_pop")
    with colP3:
        jitter_pct = st.slider(
            "Jitter continuous variables by ±%",
            0.0, 50.0, 10.0, step=1.0, key="jitter_pct"
        )

    rng = np.random.default_rng(seed)

    # ---- General scenario specification: any n variables ----
    st.markdown("**Choose which variables to change in the scenario**")

    sim_var_options = {
        "Age (years)": {
            "col": "AgeYears",
            "type": "numeric",
            "delta_min": -20.0,
            "delta_max": 20.0,
            "delta_default": -5.0,
            "step": 1.0,
            "clip_min": 18.0,
        },
        "BMI (kg/m²)": {
            "col": "bmi",
            "type": "numeric",
            "delta_min": -10.0,
            "delta_max": 10.0,
            "delta_default": -2.0,
            "step": 0.5,
            "clip_min": 15.0,
        },
        "Waist circumference (cm)": {
            "col": "waist_circumference",
            "type": "numeric",
            "delta_min": -30.0,
            "delta_max": 30.0,
            "delta_default": -5.0,
            "step": 1.0,
            "clip_min": 50.0,
        },
        "Systolic BP (mmHg)": {
            "col": "avg_systolic",
            "type": "numeric",
            "delta_min": -40.0,
            "delta_max": 40.0,
            "delta_default": -10.0,
            "step": 1.0,
            "clip_min": 80.0,
        },
        "Diastolic BP (mmHg)": {
            "col": "avg_diastolic",
            "type": "numeric",
            "delta_min": -30.0,
            "delta_max": 30.0,
            "delta_default": -5.0,
            "step": 1.0,
            "clip_min": 40.0,
        },
        "Resting HR (bpm)": {
            "col": "avg_HR",
            "type": "numeric",
            "delta_min": -30.0,
            "delta_max": 30.0,
            "delta_default": -5.0,
            "step": 1.0,
            "clip_min": 40.0,
        },
        "Income-to-poverty ratio": {
            "col": "FamIncome_to_poverty_ratio",
            "type": "numeric",
            "delta_min": -1.0,
            "delta_max": 5.0,
            "delta_default": 0.5,
            "step": 0.1,
            "clip_min": 0.1,
        },
        "Smoking status": {
            "col": "smoking",
            "type": "categorical",
        },
    }

    selected_vars = st.multiselect(
        "Variables to change in the scenario",
        list(sim_var_options.keys()),
        default=["Systolic BP (mmHg)"]
    )

    numeric_deltas = {}
    smoking_action = None

    for var_label in selected_vars:
        spec = sim_var_options[var_label]
        if spec["type"] == "numeric":
            col1_int, col2_int = st.columns([2, 1])
            with col1_int:
                delta = st.slider(
                    f"Change in {var_label}",
                    min_value=float(spec["delta_min"]),
                    max_value=float(spec["delta_max"]),
                    value=float(spec["delta_default"]),
                    step=float(spec["step"]),
                    key=f"delta_{spec['col']}"
                )
            with col2_int:
                st.write("Δ applied additively to each individual.")
            numeric_deltas[spec["col"]] = (delta, spec["clip_min"])
        elif spec["type"] == "categorical" and var_label == "Smoking status":
            smoking_action = st.selectbox(
                "Smoking change",
                ["No change", "Set all to non-smokers", "Set all to smokers"],
                index=1,
                key="smoking_action"
            )

    # ---- Stratification variable for subgroup plots ----
    st.markdown("**Stratify results by**")
    strat_choice = st.selectbox(
        "Subgroup variable for plotting",
        [
            "Gender",
            "Race",
            "Education",
            "Smoking",
            "Activity level",
            "Age group (10-year bins)",
            "BMI category",
        ],
        index=0
    )

    # ---- Run simulation ----
    if st.button("Run population simulation", key="run_sim"):
        # ---------- sample real NHANES rows ----------
        n_available = len(nhanes_sim)
        if n_available == 0:
            st.error("NHANES simulation dataset is empty or not loaded correctly.")
        else:
            idx = rng.choice(n_available, size=pop_n, replace=True)
            base_df = nhanes_sim.iloc[idx].reset_index(drop=True)

            # ---------- jitter continuous variables ----------
            base_df_model = base_df.copy()
            continuous_cols = [
                "bmi",
                "AgeYears",
                "waist_circumference",
                "avg_systolic",
                "avg_diastolic",
                "avg_HR",
                "FamIncome_to_poverty_ratio",
            ]
            if jitter_pct > 0:
                scale = jitter_pct / 100.0
                for col in continuous_cols:
                    if col in base_df_model.columns:
                        noise = rng.uniform(1.0 - scale, 1.0 + scale, size=pop_n)
                        base_df_model[col] = base_df_model[col] * noise

            # ---------- create strata labels from coded variables ----------
            # Use rounded ints for mapping to avoid issues if means were used in imputation
            def as_int_series(x):
                return pd.to_numeric(x, errors="coerce").round().astype("Int64")

            gender_label = as_int_series(base_df_model["Gender"]).map(gender_rev)
            race_label = as_int_series(base_df_model["Race"]).map(race_rev)
            educ_label = as_int_series(base_df_model["Education"]).map(education_rev)
            smoking_label = as_int_series(base_df_model["smoking"]).map(smoking_rev)
            activity_label = as_int_series(base_df_model["activity_level"]).map(activity_rev)

            strata_df = pd.DataFrame({
                "Gender": gender_label.astype(str),
                "Race": race_label.astype(str),
                "Education": educ_label.astype(str),
                "Smoking": smoking_label.astype(str),
                "Activity level": activity_label.astype(str),
                "Age group (10-year bins)": pd.cut(
                    base_df_model["AgeYears"],
                    bins=[18, 30, 40, 50, 60, 70, 90],
                    right=False,
                    include_lowest=True
                ).astype(str),
                "BMI category": pd.cut(
                    base_df_model["bmi"],
                    bins=[0, 18.5, 25, 30, 35, 100],
                    labels=["<18.5", "18.5–24.9", "25–29.9", "30–34.9", "≥35"],
                    include_lowest=True
                ).astype(str),
            })

            # Baseline predictions
            b_diab, b_ckd, b_cvd = predict_three(base_df_model)

            # ---------- apply scenario changes ----------
            int_df = base_df_model.copy()

            # Numeric deltas: additive + clipping
            for col_name, (delta, clip_min) in numeric_deltas.items():
                if col_name in int_df.columns:
                    int_df[col_name] = np.clip(int_df[col_name] + delta, clip_min, None)

            # Smoking action
            if smoking_action is not None and "Smoking status" in selected_vars:
                if smoking_action == "Set all to non-smokers":
                    int_df["smoking"] = smoke_map["No"]
                elif smoking_action == "Set all to smokers":
                    int_df["smoking"] = smoke_map["Yes"]
                # "No change" does nothing

            # Scenario predictions
            i_diab, i_ckd, i_cvd = predict_three(int_df)

            # ---------- overall summary ----------
            overall = pd.DataFrame({
                "Disease": ["Diabetes", "CKD", "CVD"],
                "Baseline_mean": [
                    float(b_diab.mean()), float(b_ckd.mean()), float(b_cvd.mean())
                ],
                "Post_mean": [
                    float(i_diab.mean()), float(i_ckd.mean()), float(i_cvd.mean())
                ],
            })
            overall["Absolute_change"] = overall["Post_mean"] - overall["Baseline_mean"]
            overall["Relative_change_%"] = np.where(
                overall["Baseline_mean"] > 0,
                100 * overall["Absolute_change"] / overall["Baseline_mean"],
                np.nan
            )

            st.markdown("**Overall average predicted risk (baseline vs scenario)**")
            st.dataframe(
                overall.style.format({
                    "Baseline_mean": "{:.5f}",
                    "Post_mean": "{:.5f}",
                    "Absolute_change": "{:.5f}",
                    "Relative_change_%": "{:.2f}",
                }),
                use_container_width=True
            )

            # ---------- % change bar chart across diseases ----------
            fig_change = go.Figure()
            fig_change.add_trace(
                go.Bar(
                    x=overall["Disease"],
                    y=overall["Relative_change_%"],
                )
            )
            fig_change.update_layout(
                yaxis_title="Relative change in mean risk (%)",
                xaxis_title="Disease",
                margin=dict(l=40, r=40, t=40, b=40),
                shapes=[
                    dict(
                        type="line",
                        x0=-0.5, x1=2.5,
                        y0=0, y1=0,
                        line=dict(color="grey", dash="dash")
                    )
                ]
            )
            st.plotly_chart(fig_change, use_container_width=True)

            # ---------- subgroup summary (side-by-side bars) ----------
            st.markdown(f"**Subgroup effects by {strat_choice}**")

            subgroup_series = strata_df[strat_choice]
            df_sub = pd.DataFrame({
                "Subgroup": subgroup_series,
                "b_diab": b_diab,
                "b_ckd": b_ckd,
                "b_cvd": b_cvd,
                "i_diab": i_diab,
                "i_ckd": i_ckd,
                "i_cvd": i_cvd,
            })

            grp = df_sub.groupby("Subgroup", as_index=False).mean()
            grp_long = pd.DataFrame({
                "Subgroup": np.repeat(grp["Subgroup"].values, 3),
                "Disease": ["Diabetes", "CKD", "CVD"] * len(grp),
                "Baseline": np.concatenate([grp["b_diab"], grp["b_ckd"], grp["b_cvd"]]),
                "Post": np.concatenate([grp["i_diab"], grp["i_ckd"], grp["i_cvd"]]),
            })
            grp_long["Absolute_change"] = grp_long["Post"] - grp_long["Baseline"]

            fig_sub = go.Figure()
            for disease in ["Diabetes", "CKD", "CVD"]:
                mask = grp_long["Disease"] == disease
                fig_sub.add_trace(
                    go.Bar(
                        x=grp_long.loc[mask, "Subgroup"],
                        y=grp_long.loc[mask, "Absolute_change"],
                        name=disease
                    )
                )

            fig_sub.update_layout(
                barmode="group",
                xaxis_title=strat_choice,
                yaxis_title="Change in mean risk (scenario - baseline)",
                margin=dict(l=40, r=40, t=40, b=40)
            )

            st.plotly_chart(fig_sub, use_container_width=True)

            st.caption(
                "Bars show the absolute change in mean modelled risk within each subgroup "
                "after applying the selected scenario, relative to baseline."
            )

