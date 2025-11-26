import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ===========================
# Global colour palette & styles
# ===========================
COLOR_DIAB = "#d62728"      # Diabetes – red
COLOR_CKD = "#1f77b4"       # CKD – dark blue
COLOR_CVD = "#4da6ff"       # CVD – light blue
COLOR_BAR = "#1f77b4"       # Default bar colour
GRID_COLOR = "rgba(0,0,0,0.08)"
ZERO_LINE_COLOR = "rgba(80,80,80,0.85)"

def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a '#rrggbb' hex color to an 'rgba(r,g,b,a)' string."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ===========================
# Global plotting style for high-res exports
# ===========================
PLOTLY_DOWNLOAD_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        # scale multiplies pixel resolution of the on-screen figure
        # scale=5 → if the chart is 1100×650 px in browser, download is 5500×3250 px
        "scale": 5,
    }
}


def style_plotly_pub(fig, width=1100, height=650,
                     title_size=30, axis_title_size=24,
                     tick_size=20, legend_size=20):
    """Apply publication-style fonts, margins, and subtle grids to a Plotly figure."""
    fig.update_layout(
        width=width,
        height=height,
        font=dict(size=tick_size),
        title=dict(
            font=dict(size=title_size),
            x=0.0,
            xanchor="left"
        ),
        legend=dict(font=dict(size=legend_size)),
        margin=dict(l=60, r=20, t=70, b=60),
    )
    fig.update_xaxes(
        title_font=dict(size=axis_title_size),
        tickfont=dict(size=tick_size),
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
    )
    fig.update_yaxes(
        title_font=dict(size=axis_title_size),
        tickfont=dict(size=tick_size),
        showgrid=True,
        gridcolor=GRID_COLOR,
        zeroline=False,
    )
    return fig

def apply_plotly_figure_editor(
    fig,
    key_prefix: str,
    default_title: str,
    default_x: str,
    default_y: str,
):
    """
    Small per-figure editor: toggle on, then adjust title/x/y labels and font sizes.
    Only used in the Researcher tools tab.
    """
    with st.expander("Figure editor", expanded=False):
        enable_edit = st.checkbox("Enable figure editor", key=f"{key_prefix}_enable")
        if enable_edit:
            title = st.text_input(
                "Figure title",
                value=default_title,
                key=f"{key_prefix}_title",
            )
            x_label = st.text_input(
                "X-axis label",
                value=default_x,
                key=f"{key_prefix}_xlabel",
            )
            y_label = st.text_input(
                "Y-axis label",
                value=default_y,
                key=f"{key_prefix}_ylabel",
            )

            title_size = st.slider(
                "Title font size",
                16, 40, 30,
                key=f"{key_prefix}_title_size",
            )
            axis_title_size = st.slider(
                "Axis title font size",
                12, 32, 24,
                key=f"{key_prefix}_axis_title_size",
            )
            tick_size = st.slider(
                "Tick label font size",
                10, 28, 20,
                key=f"{key_prefix}_tick_size",
            )
            legend_size = st.slider(
                "Legend font size",
                10, 28, 20,
                key=f"{key_prefix}_legend_size",
            )

            fig.update_layout(title=title)
            fig.update_xaxes(title_text=x_label)
            fig.update_yaxes(title_text=y_label)
            fig = style_plotly_pub(
                fig,
                title_size=title_size,
                axis_title_size=axis_title_size,
                tick_size=tick_size,
                legend_size=legend_size,
            )

    return fig

def style_altair_pub(chart, title=None, width=900, height=500):
    """Make Altair charts big, with larger fonts and faint grids for export."""
    if title is not None:
        chart = chart.properties(title=title)
    chart = chart.properties(width=width, height=height)
    chart = (
        chart
        .configure_axis(
            labelFontSize=18,
            titleFontSize=20,
            grid=True,
            gridColor="#000000",
            gridOpacity=0.08,
        )
        .configure_legend(labelFontSize=18, titleFontSize=20)
        .configure_title(fontSize=24, anchor="start")
        .configure_view(strokeWidth=0)
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
# Plotly figure editor (Research tab only)
# ===========================
def apply_plotly_figure_editor(fig, key_prefix, default_title="", default_x="", default_y=""):
    """
    Optional per-figure editor used ONLY in the Researcher tools tab.

    - User can toggle editor on/off.
    - If off: figure is returned unchanged.
    - If on: layout, fonts, bar width, zero-line, and Y-range can be adjusted.
    """
    with st.expander("Figure editor", expanded=False):
        enable = st.checkbox("Enable editor", key=f"{key_prefix}_enable")
        if not enable:
            return fig

        # Titles
        title = st.text_input("Figure title", value=default_title, key=f"{key_prefix}_title")
        x_title = st.text_input("X-axis title", value=default_x, key=f"{key_prefix}_x")
        y_title = st.text_input("Y-axis title", value=default_y, key=f"{key_prefix}_y")

        # Size
        current_width = fig.layout.width or 1100
        current_height = fig.layout.height or 650
        width = st.slider(
            "Figure width (px)", 600, 2000, int(current_width), key=f"{key_prefix}_width"
        )
        height = st.slider(
            "Figure height (px)", 400, 1400, int(current_height), key=f"{key_prefix}_height"
        )

        # Fonts
        title_size = st.slider("Title font size", 12, 40, 30, key=f"{key_prefix}_title_size")
        axis_title_size = st.slider("Axis title font size", 10, 32, 24, key=f"{key_prefix}_axis_title_size")
        tick_size = st.slider("Tick font size", 8, 28, 20, key=f"{key_prefix}_tick_size")
        legend_size = st.slider("Legend font size", 8, 28, 20, key=f"{key_prefix}_legend_size")

        # Y range
        use_custom_y = st.checkbox("Use custom Y-axis range", False, key=f"{key_prefix}_use_custom_y")
        y_min = st.number_input("Y-axis minimum", value=0.0, key=f"{key_prefix}_ymin")
        y_max = st.number_input("Y-axis maximum", value=100.0, key=f"{key_prefix}_ymax")

        # Zero line & bar width
        show_zero = st.checkbox("Show horizontal zero line (keep layout shapes)", True,
                                key=f"{key_prefix}_zero")
        bar_width = st.slider("Bar width (for bar charts)", 0.2, 1.0, 0.8,
                              key=f"{key_prefix}_bar_width")

    # ---- Apply edits ----
    fig.update_layout(
        width=width,
        height=height,
        font=dict(size=tick_size),
        title=dict(text=title, font=dict(size=title_size), x=0.0, xanchor="left"),
        legend=dict(font=dict(size=legend_size)),
    )
    fig.update_xaxes(title_text=x_title, title_font=dict(size=axis_title_size), tickfont=dict(size=tick_size))
    fig.update_yaxes(title_text=y_title, title_font=dict(size=axis_title_size), tickfont=dict(size=tick_size))

    if use_custom_y:
        fig.update_yaxes(range=[y_min, y_max])

    # tweak bar width if applicable
    for trace in fig.data:
        if isinstance(trace, go.Bar):
            trace.width = bar_width

    # hide zero-line shapes if requested
    if not show_zero:
        fig.update_layout(shapes=[])

    return fig



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
    # B. Individual sensitivity analysis
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

        # convert to % risk for display
        sens_df = pd.DataFrame({
            "Value": vals,
            "Diabetes": s_diab * 100.0,
            "CKD": s_ckd * 100.0,
            "CVD": s_cvd * 100.0,
        }).melt("Value", var_name="Disease", value_name="Risk_pct")

        sens_chart = (
            alt.Chart(sens_df)
            .mark_line(size=3)
            .encode(
                x=alt.X("Value:Q", title=sens_label_indiv),
                y=alt.Y("Risk_pct:Q", title="Predicted risk (%)"),
                color=alt.Color(
                    "Disease:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Diabetes", "CKD", "CVD"],
                        range=[COLOR_DIAB, COLOR_CKD, COLOR_CVD],
                    ),
                ),
            )
        )
        sens_chart = style_altair_pub(
            sens_chart,
            title=f"Individual sensitivity to {sens_label_indiv}"
        )
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

        # convert to % risk
        z_diab = h_diab.reshape(n_y, n_x) * 100.0
        z_ckd = h_ckd.reshape(n_y, n_x) * 100.0
        z_cvd = h_cvd.reshape(n_y, n_x) * 100.0

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
            title=(
                f"Predicted risk (%) across {heat_x_label_indiv} "
                f"and {heat_y_label_indiv} for current individual"
            ),
            coloraxis=dict(
                colorscale="Viridis",
                cmin=0.0,
                cmax=max_risk,
                colorbar=dict(title="Predicted risk (%)")
            ),
            margin=dict(l=60, r=40, t=80, b=60),
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
    # --- Terms of use gate for Researcher tools tab ---
    if "researcher_agreed" not in st.session_state:
        st.session_state["researcher_agreed"] = False

    if not st.session_state["researcher_agreed"]:
        st.markdown("### Terms of use for Researcher tools")
        st.markdown(
            """
            By using the **Researcher tools** in this app, you agree to the following:

        By using the **Researcher tools** in this app, you agree to the following:

        1. For exploratory **internal** use—such as lab discussions, coursework, or preliminary
           non-public analyses—you may use figures or tables generated by this tool, provided
           you cite:
        
           **Costa, AM and Iris, BD. Nature Scientific Reports. 2025.**
        
        2. For **any formal use** of this tool or its outputs in publications—including use in
           **manuscripts, preprints, external talks, or grant proposals**—you agree to
           **contact Arthur M. Costa (arthurcosta@uchicago.edu)** to discuss collaboration and
           **appropriate authorship**.
        
           Arthur curated the underlying dataset, engineered derived variables, and developed
           the modelling and population-simulation framework used in this tool. Guidance on
           methodology can be found in **Costa, AM and Iris, BD. Nature Scientific Reports.
           2025**, or by contacting Arthur directly.
        
        3. To request custom analyses—such as incorporation of laboratory biomarkers,
           international datasets, or specialized modelling—please contact Arthur
           (**arthurcosta@uchicago.edu**) to explore potential collaborations.
            """
        )

        with st.form("researcher_terms_form"):
            agree = st.checkbox(
                "I have read and agree to these terms.",
                key="researcher_terms_checkbox",
            )
            submitted = st.form_submit_button("Continue to Researcher tools")

        if submitted and agree:
            st.session_state["researcher_agreed"] = True
        else:
            if submitted and not agree:
                st.warning("You must agree to the terms above to use the Researcher tools.")
            # Stop rendering anything else in this tab until they agree
            st.stop()

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

            # Store for later rendering (outside the button)
            st.session_state["overall_df"] = overall



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
                # ---------- Render overall summary + chart (persists after button click) ----------
                # ---------- Render overall table + bar chart (persistent, editable) ----------
        if "overall_df" in st.session_state:
            overall = st.session_state["overall_df"]

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

            fig_change = go.Figure()
            rel_vals = overall["Relative_change_%"].values.astype(float)
            err_plus = (overall["CI_high_%"] - overall["Relative_change_%"]).values.astype(float)
            err_minus = (overall["Relative_change_%"] - overall["CI_low_%"]).values.astype(float)

            fig_change.add_trace(
                go.Bar(
                    x=overall["Disease"],
                    y=rel_vals,
                    marker=dict(color=COLOR_BAR),
                    error_y=dict(
                        type="data",
                        array=err_plus,
                        arrayminus=err_minus,
                        visible=True,
                        thickness=1.5,
                    ),
                )
            )
            fig_change.update_layout(
                title="Overall relative change in mean predicted risk",
                yaxis_title="Relative change in mean predicted risk (%)",
                xaxis_title="Disease",
                margin=dict(l=60, r=20, t=80, b=60),
                shapes=[
                    dict(
                        type="line",
                        x0=-0.5, x1=2.5,
                        y0=0, y1=0,
                        line=dict(color=ZERO_LINE_COLOR, dash="dash", width=2),
                    )
                ]
            )
            fig_change = style_plotly_pub(fig_change)

            # 🔧 Figure editor for the overall bar chart
            fig_change = apply_plotly_figure_editor(
                fig_change,
                key_prefix="overall_rel_change",
                default_title="Overall relative change in mean predicted risk",
                default_x="Disease",
                default_y="Relative change in mean predicted risk (%)",
            )

            st.plotly_chart(fig_change, use_container_width=True,
                            config=PLOTLY_DOWNLOAD_CONFIG)

       # if "overall_df" in st.session_state:
            overall = st.session_state["overall_df"]
                    # store overall summary so it can be rendered (and edited) without recomputing
        #st.session_state["overall_df"] = overall


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
                    marker=dict(color=COLOR_BAR),
                    error_y=dict(
                        type="data",
                        array=err_plus,
                        arrayminus=err_minus,
                        visible=True,
                        thickness=1.5,
                    ),
                )
            )

            fig_sub.update_layout(
                title=f"Subgroup-specific relative change in mean risk for {disease_choice}",
                barmode="group",
                xaxis_title=f"{strat_label} subgroup",
                yaxis_title=f"Relative change in mean risk for {disease_choice} (%)",
                margin=dict(l=60, r=20, t=90, b=90),
                shapes=[
                    dict(
                        type="line",
                        x0=-0.5,
                        x1=len(sub_df["Subgroup"]) - 0.5,
                        y0=0,
                        y1=0,
                        line=dict(color=ZERO_LINE_COLOR, dash="dash", width=2),
                    )
                ],
            )
            fig_sub.update_xaxes(tickangle=30)

            fig_sub = style_plotly_pub(fig_sub)
            fig_sub = apply_plotly_figure_editor(
                fig_sub,
                key_prefix="subgroup_change",
                default_title=f"Subgroup-specific relative change in mean risk for {disease_choice}",
                default_x=f"{strat_label} subgroup",
                default_y=f"Relative change in mean risk for {disease_choice} (%)",
            )
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

            # store mean and CI for each disease
            rows = []
            for v in vals:
                X_mod = base_X_pop.copy()
                X_mod[var_col] = v
                p_d, p_k, p_c = predict_three(X_mod)

                for disease_name, arr in zip(
                    ["Diabetes", "CKD", "CVD"],
                    [p_d, p_k, p_c]
                ):
                    vals_pct = arr * 100.0
                    mean = float(vals_pct.mean())
                    # within-population variance → CI of mean
                    se = float(vals_pct.std(ddof=1) / np.sqrt(len(vals_pct)))
                    ci_low = mean - 1.96 * se
                    ci_high = mean + 1.96 * se

                    rows.append({
                        "Value": v,
                        "Disease": disease_name,
                        "Mean_risk_pct": mean,
                        "CI_low": ci_low,
                        "CI_high": ci_high,
                    })

            pop_sens_df = pd.DataFrame(rows)

                        # -------- Plotly version with per-disease line + CI ribbons --------
            fig_pop_sens = go.Figure()

            color_map = {
                "Diabetes": COLOR_DIAB,
                "CKD": COLOR_CKD,
                "CVD": COLOR_CVD,
            }

            for disease_name in ["Diabetes", "CKD", "CVD"]:
                sub = pop_sens_df[pop_sens_df["Disease"] == disease_name].sort_values("Value")
                x_vals_plot = sub["Value"].values
                mean_vals = sub["Mean_risk_pct"].values
                ci_low = sub["CI_low"].values
                ci_high = sub["CI_high"].values
                col = color_map.get(disease_name, COLOR_BAR)

                # lower CI bound (invisible line)
                fig_pop_sens.add_trace(
                    go.Scatter(
                        x=x_vals_plot,
                        y=ci_low,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"{disease_name} CI low",
                    )
                )

                # upper CI bound with fill → ribbon
                fig_pop_sens.add_trace(
                    go.Scatter(
                        x=x_vals_plot,
                        y=ci_high,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonexty",
                        fillcolor=hex_to_rgba(col, 0.2),  # 20% opacity
                        showlegend=False,
                        hoverinfo="skip",
                        name=f"{disease_name} CI high",
                    )
                )

                # mean line on top
                fig_pop_sens.add_trace(
                    go.Scatter(
                        x=x_vals_plot,
                        y=mean_vals,
                        mode="lines",
                        name=disease_name,
                        line=dict(width=3, color=col),
                    )
                )

            fig_pop_sens.update_layout(
                title=f"Population mean predicted risk vs {pop_sens_label}",
                xaxis_title=pop_sens_label,
                yaxis_title="Mean predicted risk in population (%)",
                margin=dict(l=60, r=20, t=80, b=60),
            )

            fig_pop_sens = style_plotly_pub(fig_pop_sens)

            # 🔧 Figure editor toggle for this plot
            fig_pop_sens = apply_plotly_figure_editor(
                fig_pop_sens,
                key_prefix="pop_1d_sensitivity",
                default_title=f"Population mean predicted risk vs {pop_sens_label}",
                default_x=pop_sens_label,
                default_y="Mean predicted risk in population (%)",
            )

            st.plotly_chart(fig_pop_sens, use_container_width=True,
                            config=PLOTLY_DOWNLOAD_CONFIG)

            st.caption(
                "Curves show mean modelled risk in the synthetic population if everyone "
                f"had the specified value of {pop_sens_label}. Shaded ribbons are 95% CIs "
                "for the mean risk at each value."
            )


        # ------------------ Population 2D heatmaps ------------------
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

            # convert to % risk and average across population
            z_diab = (h_diab.reshape(n_grid, n_pop).mean(axis=1).reshape(n_y, n_x)) * 100.0
            z_ckd = (h_ckd.reshape(n_grid, n_pop).mean(axis=1).reshape(n_y, n_x)) * 100.0
            z_cvd = (h_cvd.reshape(n_grid, n_pop).mean(axis=1).reshape(n_y, n_x)) * 100.0

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
                title=(
                    f"Mean predicted risk (%) across {pop_heat_x_label} and "
                    f"{pop_heat_y_label} in synthetic population"
                ),
                coloraxis=dict(
                    colorscale="Viridis",
                    cmin=0.0,
                    cmax=max_risk,
                    colorbar=dict(title="Mean predicted risk (%)"),
                ),
                margin=dict(l=60, r=40, t=80, b=60),
            )

            fig_heat_pop.update_xaxes(title_text=pop_heat_x_label, row=1, col=2)
            fig_heat_pop.update_yaxes(title_text=pop_heat_y_label, row=1, col=1)
            for c in [2, 3]:
                fig_heat_pop.update_yaxes(showticklabels=False, row=1, col=c)

            fig_heat_pop = style_plotly_pub(fig_heat_pop)
            fig_heat_pop = apply_plotly_figure_editor(
                fig_heat_pop,
                key_prefix="pop_heatmaps",
                default_title=(
                    f"Mean predicted risk (%) across {pop_heat_x_label} and "
                    f"{pop_heat_y_label} in synthetic population"
                ),
                default_x=pop_heat_x_label,
                default_y=pop_heat_y_label,
            )
            
            # 🔧 Re-apply single shared axis titles so they don’t repeat on all 3 panels
            fig_heat_pop.update_xaxes(title_text="", row=1, col=1)
            fig_heat_pop.update_xaxes(title_text=pop_heat_x_label, row=1, col=2)
            fig_heat_pop.update_xaxes(title_text="", row=1, col=3)
            
            fig_heat_pop.update_yaxes(title_text=pop_heat_y_label, row=1, col=1)
            fig_heat_pop.update_yaxes(title_text="", row=1, col=2)
            fig_heat_pop.update_yaxes(title_text="", row=1, col=3)
            
            st.plotly_chart(fig_heat_pop, use_container_width=True,
                            config=PLOTLY_DOWNLOAD_CONFIG)


            st.caption(
                "Heatmaps show mean predicted risk in the synthetic population if everyone "
                "had the specified pair of values for the two selected variables."
            )

