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
pipe_ckd = joblib.load("ckd_model.joblib")
pipe_cvd = joblib.load("cvd_model.joblib")

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
education_inv_map = {v: k for k, v in education_map.items()}

gender_map = {"Male": 1, "Female": 2}
gender_inv_map = {v: k for k, v in gender_map.items()}

race_map = {
    "Non-Hispanic White": 1,
    "Non-Hispanic Black": 2,
    "Hispanic": 3,
    "Other": 4,
}
race_inv_map = {v: k for k, v in race_map.items()}

# model uses numeric smoking; here we use 0/1
smoke_map = {"No": 0, "Yes": 1}
smoke_inv_map = {v: k for k, v in smoke_map.items()}

activity_map = {"Low": 0, "Moderate": 1, "High": 2}
activity_inv_map = {v: k for k, v in activity_map.items()}

FEATURE_COLS = [
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

# Continuous vs categorical feature sets for imputation
CONTINUOUS_COLS = [
    "bmi",
    "AgeYears",
    "waist_circumference",
    "avg_systolic",
    "avg_diastolic",
    "avg_HR",
    "FamIncome_to_poverty_ratio",
]
CATEGORICAL_COLS = ["activity_level", "smoking", "Education", "Race", "Gender"]

# Allowed code ranges (contiguous in your mappings)
CAT_CODE_RANGES = {
    "activity_level": (0, 2),
    "smoking": (0, 1),
    "Education": (1, 5),
    "Race": (1, 4),
    "Gender": (1, 2),
}


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
    p_ckd = pipe_ckd.predict_proba(X)[:, 1]
    p_cvd = pipe_cvd.predict_proba(X)[:, 1]
    return p_diab, p_ckd, p_cvd


# -------------------------------------------------
# Load NHANES test set for simulation (real rows)
# -------------------------------------------------
try:
    nhanes_sim = pd.read_csv("nhanes_test_for_sim.csv")
    HAVE_NHANES_SIM = True
except Exception:
    nhanes_sim = None
    HAVE_NHANES_SIM = False

if HAVE_NHANES_SIM:
    # strip whitespace in column names
    nhanes_sim.columns = nhanes_sim.columns.str.strip()

    # ---------- recode / clean categorical variables ----------

    # Race (RIDRETH3-style -> 4-category scheme)
    # 1 = Mexican American
    # 2 = Other Hispanic
    # 3 = Non-Hispanic White
    # 4 = Non-Hispanic Black
    # 6 = Non-Hispanic Asian
    # 7 = Other / Multi
    if "Race" in nhanes_sim.columns:
        race_raw = pd.to_numeric(nhanes_sim["Race"], errors="coerce").astype("Int64")
        race_recode = pd.Series(np.nan, index=nhanes_sim.index, dtype="float")

        race_recode[race_raw == 3] = 1  # Non-Hispanic White
        race_recode[race_raw == 4] = 2  # Non-Hispanic Black
        race_recode[race_raw.isin([1, 2])] = 3  # Hispanic
        race_recode[race_raw.isin([6, 7])] = 4  # Other

        nhanes_sim["Race"] = race_recode

    # Smoking: file uses 1/2 + NHANES missing codes; standardize to 0/1
    if "smoking" in nhanes_sim.columns:
        sm_raw = pd.to_numeric(nhanes_sim["smoking"], errors="coerce").astype("Int64")
        sm_clean = pd.Series(np.nan, index=nhanes_sim.index, dtype="float")
        # 1 = non-smoker, 2 = smoker
        sm_clean[sm_raw == 1] = 0
        sm_clean[sm_raw == 2] = 1
        nhanes_sim["smoking"] = sm_clean

    # Education: keep 1–5 (your recode); other values (7,9, etc.) -> NaN
    if "Education" in nhanes_sim.columns:
        edu_raw = pd.to_numeric(nhanes_sim["Education"], errors="coerce").astype("Int64")
        edu_clean = pd.Series(np.nan, index=nhanes_sim.index, dtype="float")
        mask_edu = edu_raw.isin([1, 2, 3, 4, 5])
        edu_clean[mask_edu] = edu_raw[mask_edu]
        nhanes_sim["Education"] = edu_clean

    # Gender: keep 1/2 only; others -> NaN
    if "Gender" in nhanes_sim.columns:
        g_raw = pd.to_numeric(nhanes_sim["Gender"], errors="coerce").astype("Int64")
        g_clean = pd.Series(np.nan, index=nhanes_sim.index, dtype="float")
        mask_g = g_raw.isin([1, 2])
        g_clean[mask_g] = g_raw[mask_g]
        nhanes_sim["Gender"] = g_clean

    # Activity level: keep 0/1/2; others -> NaN
    if "activity_level" in nhanes_sim.columns:
        act_raw = pd.to_numeric(nhanes_sim["activity_level"], errors="coerce").astype("Int64")
        act_clean = pd.Series(np.nan, index=nhanes_sim.index, dtype="float")
        mask_act = act_raw.isin([0, 1, 2])
        act_clean[mask_act] = act_raw[mask_act]
        nhanes_sim["activity_level"] = act_clean

    # Force all feature cols to numeric (strings/blanks -> NaN)
    for col in FEATURE_COLS:
        if col in nhanes_sim.columns:
            nhanes_sim[col] = pd.to_numeric(nhanes_sim[col], errors="coerce")

    # Clip continuous vars to plausible ranges
    clip_ranges = {
        "AgeYears": (18, 90),
        "bmi": (15, 60),
        "waist_circumference": (50, 200),
        "avg_systolic": (80, 220),
        "avg_diastolic": (40, 140),
        "avg_HR": (40, 140),
        "FamIncome_to_poverty_ratio": (0.05, 10.0),
    }
    for col, (lo, hi) in clip_ranges.items():
        if col in nhanes_sim.columns:
            nhanes_sim[col] = nhanes_sim[col].clip(lo, hi)

    # Drop rows missing core predictors entirely
    core_cols = [c for c in ["AgeYears", "bmi", "avg_systolic", "avg_diastolic"] if c in nhanes_sim.columns]
    if core_cols:
        nhanes_sim = nhanes_sim.dropna(subset=core_cols)

    # Precompute stats for imputation
    # continuous means
    cont_existing = [c for c in CONTINUOUS_COLS if c in nhanes_sim.columns]
    nhanes_means = nhanes_sim[cont_existing].mean(numeric_only=True)

    # categorical modes (single mode per column)
    nhanes_cat_modes = {}
    for col in CATEGORICAL_COLS:
        if col in nhanes_sim.columns:
            s = nhanes_sim[col].dropna()
            if not s.empty:
                nhanes_cat_modes[col] = float(s.mode().iloc[0])
else:
    nhanes_means = pd.Series(dtype=float)
    nhanes_cat_modes = {}


def prepare_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has all feature columns.
    - Continuous: coerce to numeric, impute with NHANES mean (or column mean fallback).
    - Categorical: coerce to numeric, impute with NHANES mode (or column mode), round
      and clip to valid code ranges.
    """
    df = df.copy()

    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(df[col], errors="coerce")

        if col in CATEGORICAL_COLS:
            # mode-based imputation
            if HAVE_NHANES_SIM and col in nhanes_cat_modes:
                fill_val = nhanes_cat_modes[col]
            else:
                s = df[col].dropna()
                fill_val = float(s.mode().iloc[0]) if not s.empty else 0.0

            df[col] = df[col].fillna(fill_val)
            df[col] = df[col].round()

            # snap back into valid code range
            vmin, vmax = CAT_CODE_RANGES[col]
            df[col] = df[col].clip(vmin, vmax)

        else:
            # continuous: mean-based imputation
            if HAVE_NHANES_SIM and col in nhanes_means.index:
                fill_val = nhanes_means[col]
            else:
                fill_val = df[col].mean()

            df[col] = df[col].fillna(fill_val)

    return df[FEATURE_COLS].astype(float).copy()


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

        X_model = prepare_for_model(X)
        p_diab, p_ckd, p_cvd = predict_three(X_model)

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
        "Explore how changes in predictors affect modelled risk for "
        "diabetes, CKD, and CVD. All calculations use the same non-laboratory "
        "models as the main risk calculator."
    )

    # ---------------- Baseline profile for 1D/2D tools ----------------
    st.subheader("Baseline profile (held constant for sensitivity analyses)")
    colB1, colB2, colB3 = st.columns(3)

    with colB1:
        age_r = st.number_input("Age (years)", 18, 90, 50, key="age_r")
        bmi_r = st.number_input("BMI (kg/m²)", 15.0, 60.0, 28.0, key="bmi_r")
        waist_r = st.number_input("Waist circumference (cm)", 50.0, 200.0, 100.0, key="waist_r")

    with colB2:
        sbp_r = st.number_input("Systolic BP (mmHg)", 80, 220, 135, key="sbp_r")
        dbp_r = st.number_input("Diastolic BP (mmHg)", 40, 140, 80, key="dbp_r")
        hr_r = st.number_input("Resting heart rate (bpm)", 40, 140, 72, key="hr_r")

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
        activity_label="Moderate",  # default
        smoker_label="No",
        sbp=sbp_r, dbp=dbp_r, hr=hr_r,
        income_ratio=income_ratio_r,
        education_label=education_r,
        race_label=race_r, gender_label=gender_r
    )
    baseline_X = prepare_for_model(baseline_df)
    base_diab, base_ckd, base_cvd = predict_three(baseline_X)
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

    sens_X = pd.concat([baseline_X] * n_points, ignore_index=True)
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

    grid_X = pd.concat([baseline_X] * n_grid, ignore_index=True)
    grid_X[x_col] = X_grid.ravel()
    grid_X[y_col] = Y_grid.ravel()

    h_diab, h_ckd, h_cvd = predict_three(grid_X)

    z_diab = h_diab.reshape(n_y, n_x)
    z_ckd = h_ckd.reshape(n_y, n_x)
    z_cvd = h_cvd.reshape(n_y, n_x)

    max_risk = float(max(z_diab.max(), z_ckd.max(), z_cvd.max()))

    fig_heat = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Diabetes", "CKD", "CVD"),
        horizontal_spacing=0.06
    )

    for idx, z in enumerate([z_diab, z_ckd, z_cvd], start=1):
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
    for c in [1, 2, 3]:
        fig_heat.update_xaxes(title_text=heat_x_label, row=1, col=c)
        fig_heat.update_yaxes(title_text=heat_y_label, row=1, col=c)

    st.plotly_chart(fig_heat, use_container_width=True)
    st.caption(
        "Heatmaps show predicted risk across a 2D grid of values for the selected predictors, "
        "with all other variables fixed at the baseline profile."
    )

    st.markdown("---")

    # ============================================================
    # 3. Population scenario simulator (multi-variable)
    # ============================================================
    st.subheader("Population scenario simulator")

    if not HAVE_NHANES_SIM:
        st.warning("nhanes_test_for_sim.csv not found. Upload it to enable the simulator.")
    else:
        st.markdown(
            "Draw a synthetic population from real NHANES rows (with jitter), "
            "modify any subset of predictors, and compare baseline vs scenario risks "
            "overall and by subgroup."
        )

        colP1, colP2, colP3 = st.columns(3)
        with colP1:
            pop_n = st.slider("Population size", 500, 10000, 3000, step=500, key="pop_n")
        with colP2:
            jitter_pct = st.slider("Jitter on continuous vars (%)", 0.0, 20.0, 5.0, step=1.0)
        with colP3:
            seed = st.number_input("Random seed", min_value=0, max_value=10000, value=42, key="seed_pop")

        rng = np.random.default_rng(seed)

        # ----- numeric variables to modify (additive change) -----
        st.markdown("**Numeric variables to modify (additive change)**")
        numeric_mod_options = {
            "Age (years)": ("AgeYears", -20.0, 20.0, 1.0),
            "BMI (kg/m²)": ("bmi", -10.0, 10.0, 0.5),
            "Waist circumference (cm)": ("waist_circumference", -30.0, 30.0, 1.0),
            "Systolic BP (mmHg)": ("avg_systolic", -40.0, 40.0, 1.0),
            "Diastolic BP (mmHg)": ("avg_diastolic", -30.0, 30.0, 1.0),
            "Resting HR (bpm)": ("avg_HR", -30.0, 30.0, 1.0),
            "Income-to-poverty ratio": ("FamIncome_to_poverty_ratio", -2.0, 2.0, 0.1),
        }
        selected_numeric = st.multiselect(
            "Select numeric variables to change",
            list(numeric_mod_options.keys())
        )

        numeric_deltas = {}
        for label in selected_numeric:
            col_name, dmin, dmax, step = numeric_mod_options[label]
            numeric_deltas[col_name] = st.slider(
                f"{label} change",
                float(dmin), float(dmax), 0.0, step=float(step),
                key=f"delta_{col_name}"
            )

        # ----- categorical overrides -----
        st.markdown("**Categorical variables to override**")
        cat_mod_options = {
            "Smoking status": ("smoking", smoke_map),
            "Activity level": ("activity_level", activity_map),
            "Gender": ("Gender", gender_map),
            "Race/ethnicity": ("Race", race_map),
            "Education": ("Education", education_map),
        }
        selected_cat = st.multiselect(
            "Select categorical variables to override",
            list(cat_mod_options.keys())
        )

        cat_overrides = {}
        for label in selected_cat:
            col_name, mapping = cat_mod_options[label]
            new_label = st.selectbox(
                f"Set {label} to",
                list(mapping.keys()),
                key=f"override_{col_name}"
            )
            # cast to float to match feature dtype
            cat_overrides[col_name] = float(mapping[new_label])

        # ----- stratification options -----
        st.markdown("**Stratify results by**")
        strat_option = st.selectbox(
            "Subgroup variable",
            [
                "Gender",
                "Race",
                "Education",
                "smoking",
                "activity_level",
                "AgeYears (binned)",
                "bmi (binned)",
            ],
            index=0
        )

        if st.button("Run population scenario", key="run_sim"):
            # --- draw sample of real NHANES rows ---
            available_idx = nhanes_sim.index
            sampled_idx = rng.choice(available_idx, size=pop_n, replace=True)
            pop_df = nhanes_sim.loc[sampled_idx].copy()

            # ensure required columns exist
            for col in FEATURE_COLS:
                if col not in pop_df.columns:
                    pop_df[col] = np.nan

            # jitter continuous predictors only
            jitter_frac = jitter_pct / 100.0
            jitter_cols = CONTINUOUS_COLS

            for col in jitter_cols:
                if col not in pop_df.columns:
                    continue

                col_numeric = pd.to_numeric(pop_df[col], errors="coerce")
                mean_val = nhanes_means[col] if (HAVE_NHANES_SIM and col in nhanes_means.index) else col_numeric.mean()
                col_numeric = col_numeric.fillna(mean_val).astype(float)

                if jitter_frac > 0:
                    factor = rng.uniform(
                        1.0 - jitter_frac,
                        1.0 + jitter_frac,
                        size=col_numeric.shape[0],
                    )
                    col_numeric = col_numeric * factor

                pop_df[col] = col_numeric

            # ----- build baseline feature matrix -----
            base_X = prepare_for_model(pop_df)
            b_diab, b_ckd, b_cvd = predict_three(base_X)

            # ----- build scenario feature matrix from base_X -----
            scen_X = base_X.copy()

            # numeric deltas
            for col_name, delta in numeric_deltas.items():
                if col_name in scen_X.columns:
                    scen_X[col_name] = scen_X[col_name] + delta

            # categorical overrides
            for col_name, new_val in cat_overrides.items():
                if col_name in scen_X.columns:
                    scen_X[col_name] = new_val

            i_diab, i_ckd, i_cvd = predict_three(scen_X)

            # ---------- overall summary ----------
            overall = pd.DataFrame({
                "Disease": ["Diabetes", "CKD", "CVD"],
                "Baseline_mean": [
                    float(b_diab.mean()), float(b_ckd.mean()), float(b_cvd.mean())
                ],
                "Scenario_mean": [
                    float(i_diab.mean()), float(i_ckd.mean()), float(i_cvd.mean())
                ],
            })
            overall["Absolute_change"] = overall["Scenario_mean"] - overall["Baseline_mean"]
            overall["Relative_change_%"] = np.where(
                overall["Baseline_mean"] > 0,
                100.0 * overall["Absolute_change"] / overall["Baseline_mean"],
                np.nan
            )

            st.markdown("**Overall average predicted risk (baseline vs scenario)**")
            st.dataframe(
                overall.style.format({
                    "Baseline_mean": "{:.5f}",
                    "Scenario_mean": "{:.5f}",
                    "Absolute_change": "{:.5f}",
                    "Relative_change_%": "{:.2f}",
                }),
                use_container_width=True
            )

            # ---------- % change bar chart ----------
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

            # ---------- subgroup summary ----------
            st.markdown(f"**Subgroup effects by {strat_option}**")

            # build subgroup labels from baseline *raw* codes (pop_df)
            if strat_option == "AgeYears (binned)":
                if "AgeYears" in pop_df.columns:
                    subgroup_series = pd.cut(
                        pop_df["AgeYears"],
                        bins=[0, 40, 60, 200],
                        labels=["<40", "40–59", "≥60"]
                    )
                else:
                    subgroup_series = pd.Series(["All"] * pop_df.shape[0])
            elif strat_option == "bmi (binned)":
                if "bmi" in pop_df.columns:
                    subgroup_series = pd.cut(
                        pop_df["bmi"],
                        bins=[0, 25, 30, 100],
                        labels=["<25", "25–29.9", "≥30"]
                    )
                else:
                    subgroup_series = pd.Series(["All"] * pop_df.shape[0])
            else:
                col_name = strat_option
                if col_name not in pop_df.columns:
                    subgroup_series = pd.Series(["All"] * pop_df.shape[0])
                else:
                    codes = pd.to_numeric(pop_df[col_name], errors="coerce")
                    if col_name == "Gender":
                        subgroup_series = codes.map(gender_inv_map).fillna("Unknown")
                    elif col_name == "Race":
                        subgroup_series = codes.map(race_inv_map).fillna("Unknown")
                    elif col_name == "Education":
                        subgroup_series = codes.map(education_inv_map).fillna("Unknown")
                    elif col_name == "smoking":
                        subgroup_series = codes.map(smoke_inv_map).fillna("Unknown")
                    elif col_name == "activity_level":
                        subgroup_series = codes.map(activity_inv_map).fillna("Unknown")
                    else:
                        subgroup_series = codes.astype(str)

            df_sub = pd.DataFrame({
                "Subgroup": subgroup_series,
                "b_diab": b_diab,
                "b_ckd": b_ckd,
                "b_cvd": b_cvd,
                "i_diab": i_diab,
                "i_ckd": i_ckd,
                "i_cvd": i_cvd,
            })

            grp = df_sub.groupby("Subgroup", as_index=False).mean(numeric_only=True)
            grp_long = pd.DataFrame({
                "Subgroup": np.repeat(grp["Subgroup"].values, 3),
                "Disease": ["Diabetes", "CKD", "CVD"] * len(grp),
                "Baseline": np.concatenate([grp["b_diab"], grp["b_ckd"], grp["b_cvd"]]),
                "Scenario": np.concatenate([grp["i_diab"], grp["i_ckd"], grp["i_cvd"]]),
            })
            grp_long["Absolute_change"] = grp_long["Scenario"] - grp_long["Baseline"]

            fig_sub = go.Figure()
            for disease in ["Diabetes", "CKD", "CVD"]:
                mask = grp_long["Disease"] == disease
                fig_sub.add_trace(
                    go.Bar(
                        x=grp_long.loc[mask, "Subgroup"].astype(str),
                        y=grp_long.loc[mask, "Absolute_change"],
                        name=disease
                    )
                )

            fig_sub.update_layout(
                barmode="group",
                xaxis_title=strat_option,
                yaxis_title="Change in mean risk (scenario - baseline)",
                margin=dict(l=40, r=40, t=80, b=80),
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
            )

            st.plotly_chart(fig_sub, use_container_width=True)

            st.caption(
                "Bars show the absolute change in mean modelled risk within each subgroup "
                "after applying the chosen scenario, relative to baseline."
            )
