import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ===========================
# Global plotting style for high-res exports  🔍  NEW
# ===========================
PLOTLY_DOWNLOAD_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        # scale multiplies pixel resolution of the on-screen figure
        # scale=5 → if the chart is 1000×600 px in browser, download is 5000×3000 px
        "scale": 5,
    }
}

def style_plotly_pub(fig, width=1100, height=650,
                     title_size=26, axis_title_size=22,
                     tick_size=18, legend_size=18):
    """Apply publication-style fonts and fixed canvas size to a Plotly figure."""
    fig.update_layout(
        width=width,
        height=height,
        font=dict(size=tick_size),
        title_font=dict(size=title_size),
        legend=dict(font=dict(size=legend_size)),
    )
    fig.update_xaxes(title_font=dict(size=axis_title_size),
                     tickfont=dict(size=tick_size))
    fig.update_yaxes(title_font=dict(size=axis_title_size),
                     tickfont=dict(size=tick_size))
    return fig


def style_altair_pub(chart, title=None, width=900, height=500):
    """Make Altair charts big with larger fonts so their PNG exports look good."""
    if title is not None:
        chart = chart.properties(title=title)
    chart = chart.properties(width=width, height=height)
    chart = (
        chart
        .configure_axis(labelFontSize=16, titleFontSize=18)
        .configure_legend(labelFontSize=16, titleFontSize=18)
        .configure_title(fontSize=22)
    )
    return chart


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

# model trained with numeric “smoking”; we use 0/1 here
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

# IMPORTANT: these are treated as categorical inside the training pipeline
forced_cats = {"Gender", "Race", "Education", "smoking"}

# disease ↔ columns mapping for subgroup plotting
DISEASE_LABELS = ["Diabetes", "CKD", "CVD"]
DISEASE_TO_COLS = {
    "Diabetes": ("b_diab", "i_diab"),
    "CKD": ("b_ckd", "i_ckd"),
    "CVD": ("b_cvd", "i_cvd"),
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


def bootstrap_rel_change(b_vals, i_vals, B=300, rng=None):
    """
    Bootstrap 95% CI for *relative* (%) change in mean risk:
    100 * (mean_scenario - mean_baseline) / mean_baseline
    """
    b_vals = np.asarray(b_vals, float)
    i_vals = np.asarray(i_vals, float)

    mask = np.isfinite(b_vals) & np.isfinite(i_vals)
    b_vals = b_vals[mask]
    i_vals = i_vals[mask]
    n = len(b_vals)
    if n == 0:
        return np.nan, np.nan

    if rng is None:
        rng = np.random.default_rng()

    idx = rng.integers(0, n, size=(B, n))
    b_means = b_vals[idx].mean(axis=1)
    i_means = i_vals[idx].mean(axis=1)
    rel_changes = np.where(
        b_means > 0,
        100.0 * (i_means - b_means) / b_means,
        np.nan
    )
    lo, hi = np.nanpercentile(rel_changes, [2.5, 97.5])
    return float(lo), float(hi)


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

    # Race (RIDRETH3 -> 4-category scheme)
    if "Race" in nhanes_sim.columns:
        race_raw = pd.to_numeric(nhanes_sim["Race"], errors="coerce").astype("Int64")
        race_recode = pd.Series(np.nan, index=nhanes_sim.index, dtype="float")

        race_recode[race_raw == 3] = 1  # Non-Hispanic White
        race_recode[race_raw == 4] = 2  # Non-Hispanic Black
        race_recode[race_raw.isin([1, 2])] = 3  # Hispanic
        race_recode[race_raw.isin([6, 7])] = 4  # Other

        nhanes_sim["Race"] = race_recode

    # Smoking: original file had 1/2; standardize to 0/1
    if "smoking" in nhanes_sim.columns:
        sm_raw = pd.to_numeric(nhanes_sim["smoking"], errors="coerce").astype("Int64")
        sm_clean = pd.Series(np.nan, index=nhanes_sim.index, dtype="float")
        # assume 1 = non-smoker, 2 = smoker
        sm_clean[sm_raw == 1] = 0
        sm_clean[sm_raw == 2] = 1
        nhanes_sim["smoking"] = sm_clean

    # Education: keep 1–5, others -> NaN
    if "Education" in nhanes_sim.columns:
        edu_raw = pd.to_numeric(nhanes_sim["Education"], errors="coerce").astype("Int64")
        edu_clean = pd.Series(np.nan, index=nhanes_sim.index, dtype="float")
        edu_clean[edu_raw.isin([1, 2, 3, 4, 5])] = edu_raw[edu_raw.isin([1, 2, 3, 4, 5])]
        nhanes_sim["Education"] = edu_clean

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

    # Precompute means ONLY for numeric predictors
    numeric_for_mean = [c for c in FEATURE_COLS if c not in forced_cats]
    existing_cols = [c for c in numeric_for_mean if c in nhanes_sim.columns]
    nhanes_means = nhanes_sim[existing_cols].mean(numeric_only=True)
else:
    nhanes_means = pd.Series(dtype=float)


def prepare_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has all feature columns, coerce to numeric, and
    impute missing values ONLY for numeric predictors.
    """
    df = df.copy()

    # ensure all feature columns exist and are numeric
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = [c for c in FEATURE_COLS if c not in forced_cats]
    cat_cols = [c for c in FEATURE_COLS if c in forced_cats]

    # numeric: impute with NHANES means
    for col in num_cols:
        mean_val = nhanes_means.get(col, df[col].mean())
        df[col] = df[col].fillna(mean_val)

    # categorical: DO NOT mean-impute; leave as codes/NaNs
    for col in cat_cols:
        df[col] = df[col].round()

    return df[FEATURE_COLS].copy()


# ===========================
# Tabs
# ===========================
tab_calc, tab_research = st.tabs(["Risk calculator", "Researcher tools"])


# ============================================================
#                     TAB 1: RISK CALCULATOR
# ============================================================
with tab_calc:
    # inner tabs: individual calculator vs individual sensitivity
    tab_indiv_calc, tab_indiv_sens = st.tabs(
        ["Individual risk calculator", "Individual sensitivity analysis"]
    )

    # ---------------------------
    # A. Individual risk calculator
    # ---------------------------
    with tab_indiv_calc:
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

    # ---------------------------
    # B. Individual sensitivity analysis (lightly edited)
    # ---------------------------
    with tab_indiv_sens:
        st.title("Individual sensitivity analysis")
        st.caption(
            "These plots use the **same profile** you entered in the "
            "‘Individual risk calculator’ tab above as the baseline individual."
        )

        baseline_df = build_feature_df(
            bmi=bmi, age=age, waist=waist,
            activity_label=activity,
            smoker_label=smoker,
            sbp=sbp, dbp=dbp, hr=hr,
            income_ratio=income_ratio,
            education_label=education,
            race_label=race, gender_label=gender
        )
        baseline_X = prepare_for_model(baseline_df)
        base_diab, base_ckd, base_cvd = predict_three(baseline_X)

        st.markdown(
            f"Baseline predicted risks – Diabetes: **{base_diab[0]*100:.2f}%**, "
            f"CKD: **{base_ckd[0]*100:.2f}%**, CVD: **{base_cvd[0]*100:.2f}%**."
        )

        st.markdown("---")

        # 1D sensitivity (individual)
        st.subheader("One-dimensional sensitivity analysis (individual)")

        var_options_indiv = {
            "Age (years)": ("AgeYears", 20, 85, 40),
            "BMI (kg/m²)": ("bmi", 18, 45, 40),
            "Waist circumference (cm)": ("waist_circumference", 60, 140, 40),
            "Systolic BP (mmHg)": ("avg_systolic", 90, 180, 40),
            "Diastolic BP (mmHg)": ("avg_diastolic", 50, 110, 40),
            "Resting HR (bpm)": ("avg_HR", 50, 110, 40),
            "Income-to-poverty ratio": ("FamIncome_to_poverty_ratio", 0.3, 5.0, 40),
        }

        sens_label_indiv = st.selectbox(
            "Choose variable to vary",
            list(var_options_indiv.keys()),
            index=0,
            key="indiv_sens_var"
        )
        var_col, vmin, vmax, n_points = var_options_indiv[sens_label_indiv]
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
                x=alt.X("Value:Q", title=sens_label_indiv),
                y=alt.Y("Risk:Q", title="Predicted risk"),
                color=alt.Color("Disease:N", title=None),
            )
        )
        sens_chart = style_altair_pub(sens_chart)
        st.altair_chart(sens_chart, use_container_width=True)
        st.caption("Curves vary one predictor for your current profile, holding all others fixed.")

        st.markdown("---")

        # 2D heatmaps (individual)
        st.subheader("Two-variable interaction heatmaps (individual)")

        two_d_var_options_indiv = {
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
            heat_x_label_indiv = st.selectbox(
                "X-axis variable",
                list(two_d_var_options_indiv.keys()),
                index=3,
                key="indiv_heat_x"
            )
        with colH2:
            y_choices = [k for k in two_d_var_options_indiv.keys() if k != heat_x_label_indiv]
            heat_y_label_indiv = st.selectbox(
                "Y-axis variable",
                y_choices,
                index=1,
                key="indiv_heat_y"
            )

        (x_col, x_min, x_max) = two_d_var_options_indiv[heat_x_label_indiv]
        (y_col, y_min, y_max) = two_d_var_options_indiv[heat_y_label_indiv]

        n_x = st.slider("Resolution (X)", 5, 40, 20, key="indiv_nx_heat")
        n_y = st.slider("Resolution (Y)", 5, 40, 20, key="indiv_ny_heat")

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

        fig_heat.update_xaxes(title_text=heat_x_label_indiv, row=1, col=2)
        fig_heat.update_yaxes(title_text=heat_y_label_indiv, row=1, col=1)
        for c in [2, 3]:
            fig_heat.update_yaxes(showticklabels=False, row=1, col=c)

        fig_heat = style_plotly_pub(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True,
                        config=PLOTLY_DOWNLOAD_CONFIG)
        st.caption(
            "Heatmaps show predicted risk for your current profile across a 2D grid "
            "of values for the selected predictors."
        )


# ============================================================
#                 TAB 2: RESEARCHER TOOLS (POPULATION-BASED)
# ============================================================
with tab_research:
    st.title("Researcher tools & population sensitivity")
    st.caption(
        "All tools in this tab use a synthetic population sampled from NHANES. "
        "You can modify predictors, compare baseline vs scenario risks, and explore "
        "population-level sensitivity curves and heatmaps."
    )

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
            cat_overrides[col_name] = mapping[new_label]

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

            # jitter continuous predictors
            jitter_frac = jitter_pct / 100.0
            jitter_cols = [
                "AgeYears",
                "bmi",
                "waist_circumference",
                "avg_systolic",
                "avg_diastolic",
                "avg_HR",
                "FamIncome_to_poverty_ratio",
            ]

            for col in jitter_cols:
                if col not in pop_df.columns:
                    continue

                col_numeric = pd.to_numeric(pop_df[col], errors="coerce")
                mean_val = nhanes_means.get(col, col_numeric.mean())
                col_numeric = col_numeric.fillna(mean_val).astype(float)

                if jitter_frac > 0:
                    factor = rng.uniform(
                        1.0 - jitter_frac,
                        1.0 + jitter_frac,
                        size=col_numeric.shape[0],
                    )
                    col_numeric = col_numeric * factor

                pop_df[col] = col_numeric

            # ----- build baseline feature matrix from raw pop_df -----
            base_X = prepare_for_model(pop_df)
            b_diab, b_ckd, b_cvd = predict_three(base_X)

            # ----- build scenario dataframe from raw pop_df -----
            scenario_df = pop_df.copy()

            # numeric deltas
            for col_name, delta in numeric_deltas.items():
                if col_name in scenario_df.columns:
                    scenario_df[col_name] = scenario_df[col_name] + delta

            # categorical overrides
            for col_name, new_val in cat_overrides.items():
                if col_name in scenario_df.columns:
                    scenario_df[col_name] = new_val

            scen_X = prepare_for_model(scenario_df)
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

            # Bootstrap 95% CIs for relative change (%)
            ci_diab = bootstrap_rel_change(b_diab, i_diab, B=300, rng=rng)
            ci_ckd = bootstrap_rel_change(b_ckd, i_ckd, B=300, rng=rng)
            ci_cvd = bootstrap_rel_change(b_cvd, i_cvd, B=300, rng=rng)
            overall["CI_low_%"] = [ci_diab[0], ci_ckd[0], ci_cvd[0]]
            overall["CI_high_%"] = [ci_diab[1], ci_ckd[1], ci_cvd[1]]

            st.markdown("**Overall average predicted risk (baseline vs scenario)**")
            st.dataframe(
                overall.style.format({
                    "Baseline_mean": "{:.5f}",
                    "Scenario_mean": "{:.5f}",
                    "Absolute_change": "{:.5f}",
                    "Relative_change_%": "{:.2f}",
                    "CI_low_%": "{:.2f}",
                    "CI_high_%": "{:.2f}",
                }),
                use_container_width=True
            )

            # ---------- % change bar chart with CIs ----------
            fig_change = go.Figure()
            rel_vals = overall["Relative_change_%"].values.astype(float)
            err_plus = (overall["CI_high_%"] - overall["Relative_change_%"]).values.astype(float)
            err_minus = (overall["Relative_change_%"] - overall["CI_low_%"]).values.astype(float)

            fig_change.add_trace(
                go.Bar(
                    x=overall["Disease"],
                    y=rel_vals,
                    error_y=dict(
                        type="data",
                        array=err_plus,
                        arrayminus=err_minus,
                        visible=True,
                    ),
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
            fig_change = style_plotly_pub(fig_change)
            st.plotly_chart(fig_change, use_container_width=True,
                            config=PLOTLY_DOWNLOAD_CONFIG)

            # ---------- subgroup summary ----------
            st.markdown(f"**Subgroup data prepared for stratification by {strat_option}**")

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
                    codes = pop_df[col_name]
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

            st.session_state["df_sub"] = df_sub
            st.session_state["strat_option"] = strat_option
            st.session_state["base_X"] = base_X

            st.caption(
                "Bars above show overall relative (%) change in mean modelled risk. "
                "Use the controls below to examine how different subgroups and global shifts "
                "affect these changes."
            )

        # ------------------ Subgroup contributions plot ------------------
        if "df_sub" in st.session_state:
            df_sub = st.session_state["df_sub"]
            strat_label = st.session_state["strat_option"]

            st.markdown(f"**Subgroup contributions to overall change – grouped by {strat_label}**")

            disease_choice = st.selectbox(
                "Disease to visualize by subgroup",
                DISEASE_LABELS,
                key="subgroup_disease_choice"
            )

            b_col, i_col = DISEASE_TO_COLS[disease_choice]

            rows = []
            rng_local = np.random.default_rng(seed + 12345)
            for subgroup_name, sub_group_df in df_sub.groupby("Subgroup"):
                b_vals = sub_group_df[b_col].values
                i_vals = sub_group_df[i_col].values

                base_mean = np.nanmean(b_vals)
                scen_mean = np.nanmean(i_vals)
                if base_mean > 0:
                    rel_change = 100.0 * (scen_mean - base_mean) / base_mean
                else:
                    rel_change = np.nan

                ci_low, ci_high = bootstrap_rel_change(
                    b_vals, i_vals, B=300, rng=rng_local
                )

                rows.append({
                    "Subgroup": str(subgroup_name),
                    "Rel_change_%": rel_change,
                    "CI_low_%": ci_low,
                    "CI_high_%": ci_high,
                })

            sub_df = pd.DataFrame(rows).sort_values("Subgroup")

            fig_sub = go.Figure()
            rel_vals = sub_df["Rel_change_%"].values.astype(float)
            err_plus = (sub_df["CI_high_%"] - sub_df["Rel_change_%"]).values.astype(float)
            err_minus = (sub_df["Rel_change_%"] - sub_df["CI_low_%"]).values.astype(float)

            fig_sub.add_trace(
                go.Bar(
                    x=sub_df["Subgroup"],
                    y=rel_vals,
                    error_y=dict(
                        type="data",
                        array=err_plus,
                        arrayminus=err_minus,
                        visible=True,
                    ),
                )
            )

            fig_sub.update_layout(
                barmode="group",
                xaxis_title=f"{strat_label} subgroup",
                yaxis_title=f"Relative change in mean risk for {disease_choice} (%)",
                margin=dict(l=40, r=40, t=80, b=80),
            )

            fig_sub = style_plotly_pub(fig_sub)
            st.plotly_chart(fig_sub, use_container_width=True,
                            config=PLOTLY_DOWNLOAD_CONFIG)

            st.caption(
                (
                    "Bars show how each subgroup contributes to the overall relative (%) change "
                    "in mean modelled risk for {disease_choice}. The sum of the subgroup effects "
                    "is consistent with the overall relative change shown above (up to sampling error)."
                ).format(disease_choice=disease_choice)
            )

        # ------------------ Population 1D sensitivity ------------------
        if "base_X" in st.session_state:
            st.markdown("---")
            st.subheader("Population-level one-dimensional sensitivity")

            base_X_pop = st.session_state["base_X"]

            pop_var_options = {
                "Age (years)": ("AgeYears", 20, 85, 40),
                "BMI (kg/m²)": ("bmi", 18, 45, 40),
                "Waist circumference (cm)": ("waist_circumference", 60, 140, 40),
                "Systolic BP (mmHg)": ("avg_systolic", 90, 180, 40),
                "Diastolic BP (mmHg)": ("avg_diastolic", 50, 110, 40),
                "Resting HR (bpm)": ("avg_HR", 50, 110, 40),
                "Income-to-poverty ratio": ("FamIncome_to_poverty_ratio", 0.3, 5.0, 40),
            }

            pop_sens_label = st.selectbox(
                "Variable to shift across the whole population",
                list(pop_var_options.keys()),
                index=0,
                key="pop_sens_var"
            )
            var_col, vmin, vmax, n_points = pop_var_options[pop_sens_label]
            vals = np.linspace(vmin, vmax, n_points)

            mean_diab = []
            mean_ckd = []
            mean_cvd = []

            for v in vals:
                X_mod = base_X_pop.copy()
                X_mod[var_col] = v
                p_d, p_k, p_c = predict_three(X_mod)
                mean_diab.append(p_d.mean())
                mean_ckd.append(p_k.mean())
                mean_cvd.append(p_c.mean())

            pop_sens_df = pd.DataFrame({
                "Value": vals,
                "Diabetes": mean_diab,
                "CKD": mean_ckd,
                "CVD": mean_cvd,
            }).melt("Value", var_name="Disease", value_name="Mean_risk")

            pop_sens_chart = (
                alt.Chart(pop_sens_df)
                .mark_line()
                .encode(
                    x=alt.X("Value:Q", title=pop_sens_label),
                    y=alt.Y("Mean_risk:Q", title="Mean predicted risk in population"),
                    color=alt.Color("Disease:N", title=None),
                )
            )
            pop_sens_chart = style_altair_pub(pop_sens_chart)
            st.altair_chart(pop_sens_chart, use_container_width=True)
            st.caption(
                "Curves show mean modelled risk in the synthetic population if everyone "
                f"had the specified value of {pop_sens_label}."
            )

        # ------------------ Population 2D heatmaps (fixed) ------------------
        if "base_X" in st.session_state:
            st.markdown("---")
            st.subheader("Population-level two-variable heatmaps")

            base_X_pop_full = st.session_state["base_X"]
            n_pop_total = base_X_pop_full.shape[0]
            subset_n = min(500, n_pop_total)
            base_X_pop = base_X_pop_full.sample(subset_n, random_state=seed).reset_index(drop=True)
            n_pop = base_X_pop.shape[0]

            pop_two_d_var_options = {
                "Age (years)": ("AgeYears", 20, 85),
                "BMI (kg/m²)": ("bmi", 18, 45),
                "Waist circumference (cm)": ("waist_circumference", 60, 140),
                "Systolic BP (mmHg)": ("avg_systolic", 90, 180),
                "Diastolic BP (mmHg)": ("avg_diastolic", 50, 110),
                "Resting HR (bpm)": ("avg_HR", 50, 110),
                "Income-to-poverty ratio": ("FamIncome_to_poverty_ratio", 0.3, 5.0),
            }

            colH1p, colH2p = st.columns(2)
            with colH1p:
                pop_heat_x_label = st.selectbox(
                    "X-axis variable",
                    list(pop_two_d_var_options.keys()),
                    index=3,
                    key="pop_heat_x"
                )
            with colH2p:
                y_choices = [k for k in pop_two_d_var_options.keys() if k != pop_heat_x_label]
                pop_heat_y_label = st.selectbox(
                    "Y-axis variable",
                    y_choices,
                    index=1,
                    key="pop_heat_y"
                )

            (x_col, x_min, x_max) = pop_two_d_var_options[pop_heat_x_label]
            (y_col, y_min, y_max) = pop_two_d_var_options[pop_heat_y_label]

            n_x = st.slider("Resolution (X)", 10, 60, 25, key="pop_nx_heat")
            n_y = st.slider("Resolution (Y)", 10, 60, 25, key="pop_ny_heat")

            x_vals = np.linspace(x_min, x_max, n_x)
            y_vals = np.linspace(y_min, y_max, n_y)
            X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
            n_grid = X_grid.size

            grid_X = pd.concat([base_X_pop] * n_grid, ignore_index=True)
            grid_X[x_col] = np.repeat(X_grid.ravel(), n_pop)
            grid_X[y_col] = np.repeat(Y_grid.ravel(), n_pop)

            h_diab, h_ckd, h_cvd = predict_three(grid_X)

            z_diab = h_diab.reshape(n_grid, n_pop).mean(axis=1).reshape(n_y, n_x)
            z_ckd = h_ckd.reshape(n_grid, n_pop).mean(axis=1).reshape(n_y, n_x)
            z_cvd = h_cvd.reshape(n_grid, n_pop).mean(axis=1).reshape(n_y, n_x)

            max_risk = float(max(z_diab.max(), z_ckd.max(), z_cvd.max()))

            fig_heat_pop = make_subplots(
                rows=1, cols=3,
                subplot_titles=("Diabetes", "CKD", "CVD"),
                horizontal_spacing=0.06
            )

            for idx, z in enumerate([z_diab, z_ckd, z_cvd], start=1):
                fig_heat_pop.add_trace(
                    go.Heatmap(
                        x=x_vals,
                        y=y_vals,
                        z=z,
                        coloraxis="coloraxis"
                    ),
                    row=1, col=idx
                )

            fig_heat_pop.update_layout(
                coloraxis=dict(colorscale="Viridis", cmin=0.0, cmax=max_risk),
                margin=dict(l=40, r=40, t=40, b=40),
            )

            fig_heat_pop.update_xaxes(title_text=pop_heat_x_label, row=1, col=2)
            fig_heat_pop.update_yaxes(title_text=pop_heat_y_label, row=1, col=1)
            for c in [2, 3]:
                fig_heat_pop.update_yaxes(showticklabels=False, row=1, col=c)

            fig_heat_pop = style_plotly_pub(fig_heat_pop)
            st.plotly_chart(fig_heat_pop, use_container_width=True,
                            config=PLOTLY_DOWNLOAD_CONFIG)
            st.caption(
                "Heatmaps show mean predicted risk in the synthetic population if everyone "
                "had the specified pair of values for the two selected variables."
            )

