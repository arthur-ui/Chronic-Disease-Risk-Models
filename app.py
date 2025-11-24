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
    # 3. Population intervention simulator
    # ============================================================
    st.subheader("Population intervention simulator")

    st.markdown(
        "Simulate a synthetic population, apply a single intervention, and "
        "compare predicted risk before and after the intervention overall and "
        "across subgroups."
    )

    colP1, colP2 = st.columns(2)
    with colP1:
        pop_n = st.slider("Population size", 500, 20000, 3000, step=500, key="pop_n")
    with colP2:
        seed = st.number_input("Random seed", min_value=0, max_value=10000, value=42, key="seed_pop")

    rng = np.random.default_rng(seed)

    intervention = st.selectbox(
        "Choose intervention",
        [
            "Lower systolic BP",
            "Lower diastolic BP",
            "Reduce BMI",
            "Reduce waist circumference",
            "Increase income-to-poverty ratio",
            "Set all smokers to non-smokers",
        ],
        index=0
    )

    delta_val = None
    if intervention in [
        "Lower systolic BP",
        "Lower diastolic BP",
        "Reduce BMI",
        "Reduce waist circumference",
        "Increase income-to-poverty ratio",
    ]:
        label = {
            "Lower systolic BP": "SBP reduction (mmHg)",
            "Lower diastolic BP": "DBP reduction (mmHg)",
            "Reduce BMI": "BMI reduction (kg/m²)",
            "Reduce waist circumference": "Waist reduction (cm)",
            "Increase income-to-poverty ratio": "Increase in income-to-poverty ratio",
        }[intervention]
        default = {
            "Lower systolic BP": 10.0,
            "Lower diastolic BP": 5.0,
            "Reduce BMI": 2.0,
            "Reduce waist circumference": 5.0,
            "Increase income-to-poverty ratio": 0.5,
        }[intervention]
        max_val = {
            "Lower systolic BP": 40.0,
            "Lower diastolic BP": 30.0,
            "Reduce BMI": 10.0,
            "Reduce waist circumference": 30.0,
            "Increase income-to-poverty ratio": 5.0,
        }[intervention]
        step = 1.0 if "BP" in intervention or "Waist" in intervention else 0.1
        delta_val = st.slider(label, 0.0, float(max_val), float(default), step=float(step), key="delta_int")

    subgroup_var = st.selectbox(
        "Subgroup to stratify by",
        ["Gender", "Race", "Education"]
    )

    if st.button("Run population simulation", key="run_sim"):
        # ---------- generate synthetic population ----------
        age_pop = np.clip(rng.normal(loc=50, scale=15, size=pop_n), 18, 90)
        bmi_pop = np.clip(rng.normal(loc=28, scale=5, size=pop_n), 18, 45)
        waist_pop = np.clip(rng.normal(loc=100, scale=15, size=pop_n), 60, 150)
        sbp_pop = np.clip(rng.normal(loc=130, scale=15, size=pop_n), 90, 200)
        dbp_pop = np.clip(rng.normal(loc=80, scale=10, size=pop_n), 50, 120)
        hr_pop  = np.clip(rng.normal(loc=72, scale=10, size=pop_n), 50, 110)
        income_ratio_pop = np.clip(
            rng.lognormal(mean=np.log(2.0), sigma=0.5, size=pop_n),
            0.3, 6.0
        )

        gender_pop = rng.choice(["Male", "Female"], size=pop_n, p=[0.48, 0.52])
        race_pop = rng.choice(
            ["Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Other"],
            size=pop_n,
            p=[0.6, 0.15, 0.18, 0.07]
        )
        educ_keys = list(education_map.keys())
        educ_pop = rng.choice(
            educ_keys,
            size=pop_n,
            p=[0.05, 0.10, 0.25, 0.30, 0.30]
        )
        smoking_pop = rng.choice(["No", "Yes"], size=pop_n, p=[0.6, 0.4])
        activity_pop = rng.choice(["Low", "Moderate", "High"], size=pop_n, p=[0.3, 0.5, 0.2])

        base_df = pd.DataFrame({
            "bmi": bmi_pop,
            "AgeYears": age_pop,
            "waist_circumference": waist_pop,
            "activity_level": [activity_map[a] for a in activity_pop],
            "smoking": [smoke_map[s] for s in smoking_pop],
            "avg_systolic": sbp_pop,
            "avg_diastolic": dbp_pop,
            "avg_HR": hr_pop,
            "FamIncome_to_poverty_ratio": income_ratio_pop,
            "Education": [education_map[e] for e in educ_pop],
            "Race": [race_map[r] for r in race_pop],
            "Gender": [gender_map[g] for g in gender_pop],
        })

        base_df_human = pd.DataFrame({
            "Gender": gender_pop,
            "Race": race_pop,
            "Education": educ_pop,
        })

        b_diab, b_ckd, b_cvd = predict_three(base_df)

        # ---------- apply intervention ----------
        int_df = base_df.copy()

        if intervention == "Lower systolic BP" and delta_val is not None:
            int_df["avg_systolic"] = np.clip(int_df["avg_systolic"] - delta_val, 80, None)
        elif intervention == "Lower diastolic BP" and delta_val is not None:
            int_df["avg_diastolic"] = np.clip(int_df["avg_diastolic"] - delta_val, 40, None)
        elif intervention == "Reduce BMI" and delta_val is not None:
            int_df["bmi"] = np.clip(int_df["bmi"] - delta_val, 15, None)
        elif intervention == "Reduce waist circumference" and delta_val is not None:
            int_df["waist_circumference"] = np.clip(
                int_df["waist_circumference"] - delta_val, 50, None
            )
        elif intervention == "Increase income-to-poverty ratio" and delta_val is not None:
            int_df["FamIncome_to_poverty_ratio"] = int_df["FamIncome_to_poverty_ratio"] + delta_val
        elif intervention == "Set all smokers to non-smokers":
            int_df["smoking"] = smoke_map["No"]

        i_diab, i_ckd, i_cvd = predict_three(int_df)

        # --- debug readouts for BP / smoking interventions ---
        if intervention == "Lower systolic BP":
            st.write(
                f"Mean SBP before: {base_df['avg_systolic'].mean():.2f}, "
                f"after: {int_df['avg_systolic'].mean():.2f}"
            )
        elif intervention == "Lower diastolic BP":
            st.write(
                f"Mean DBP before: {base_df['avg_diastolic'].mean():.2f}, "
                f"after: {int_df['avg_diastolic'].mean():.2f}"
            )
        elif intervention == "Set all smokers to non-smokers":
            st.write(
                f"Proportion smokers before: {base_df['smoking'].mean():.3f}, "
                f"after: {int_df['smoking'].mean():.3f}"
            )

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

        st.markdown("**Overall average predicted risk (baseline vs post-intervention)**")
        st.dataframe(
            overall.style.format({
                "Baseline_mean": "{:.5f}",
                "Post_mean": "{:.5f}",
                "Absolute_change": "{:.5f}",
                "Relative_change_%": "{:.2f}",
            }),
            use_container_width=True
        )

        # ---------- % change bar chart (replaces histogram) ----------
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
        st.markdown(f"**Subgroup effects by {subgroup_var}**")

        subgroup_series = base_df_human[subgroup_var]
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

        # Plotly grouped bar chart: one cluster per subgroup, 3 bars (diseases)
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
            xaxis_title=subgroup_var,
            yaxis_title="Change in mean risk (post - baseline)",
            margin=dict(l=40, r=40, t=40, b=40)
        )

        st.plotly_chart(fig_sub, use_container_width=True)

        st.caption(
            "Bars show the absolute change in mean modelled risk within each subgroup "
            "after applying the selected intervention, relative to baseline."
        )
