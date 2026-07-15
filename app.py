import os
import json
import datetime

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gspread
from google.oauth2.service_account import Credentials

# ---------- IRB-safe binning helpers ----------

def bin_age(age: float) -> str:
    if age < 30:
        return "<30"
    elif age < 45:
        return "30–44"
    elif age < 60:
        return "45–59"
    elif age < 75:
        return "60–74"
    else:
        return "75+"

def bin_bmi(bmi: float) -> str:
    if bmi < 18.5:
        return "<18.5"
    elif bmi < 25:
        return "18.5–24.9"
    elif bmi < 30:
        return "25–29.9"
    else:
        return ">=30"

def bin_waist(waist_cm: float) -> str:
    if waist_cm < 80:
        return "<80"
    elif waist_cm < 95:
        return "80–94"
    elif waist_cm < 110:
        return "95–109"
    else:
        return ">=110"

def bin_sbp(sbp: float) -> str:
    if sbp < 110:
        return "<110"
    elif sbp < 130:
        return "110–129"
    elif sbp < 160:
        return "130–159"
    else:
        return ">=160"

def bin_dbp(dbp: float) -> str:
    if dbp < 70:
        return "<70"
    elif dbp < 80:
        return "70–79"
    elif dbp < 90:
        return "80–89"
    else:
        return ">=90"

def bin_hr(hr: float) -> str:
    if hr < 60:
        return "<60"
    elif hr < 75:
        return "60–74"
    elif hr < 90:
        return "75–89"
    else:
        return ">=90"

def bin_income_ratio(r: float) -> str:
    if r < 1.0:
        return "<1.0"
    elif r < 2.0:
        return "1.0–1.99"
    elif r < 4.0:
        return "2.0–3.99"
    else:
        return ">=4.0"


# 🔧 Add this class so joblib can unpickle your models
class CalibratedPipeline:
    """Wraps a fitted sklearn Pipeline (with predict_proba)
    and an IsotonicRegression calibrator."""
    def __init__(self, base, calibrator):
        self.base = base
        self.calibrator = calibrator

    def predict_proba(self, X):
        # base probabilities
        p = self.base.predict_proba(X)[:, 1]
        # calibrated probabilities
        p_cal = self.calibrator.predict(p)
        p_cal = np.clip(p_cal, 0.0, 1.0)
        return np.vstack([1 - p_cal, p_cal]).T

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] > threshold).astype(int)


# ===========================
# Global colour palette & styles
# ===========================
COLOR_DIAB = "#1f77b4"   # Diabetes – blue
COLOR_CKD  = "#ff7f0e"   # CKD – orange
COLOR_CVD  = "#2ca02c"   # CVD – green
COLOR_BAR  = "#1f77b4"   # Default bar colour (optional)
GRID_COLOR = "rgba(0,0,0,0.08)"
ZERO_LINE_COLOR = "rgba(80,80,80,0.85)"


# ===========================
# Country list
# ===========================
COUNTRY_LIST = [
    "Prefer not to say",
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda",
    "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain",
    "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
    "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
    "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada",
    "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros",
    "Congo (Brazzaville)", "Congo (Kinshasa)", "Costa Rica", "Croatia", "Cuba",
    "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica",
    "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea",
    "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France",
    "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada",
    "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras",
    "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland",
    "Israel", "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya",
    "Kiribati", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon", "Lesotho",
    "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar",
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands",
    "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco",
    "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar", "Namibia",
    "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger",
    "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan",
    "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines",
    "Poland", "Portugal", "Qatar", "Romania", "Russia", "Rwanda",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines",
    "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal",
    "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia",
    "Solomon Islands", "Somalia", "South Africa", "South Korea", "South Sudan",
    "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria",
    "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo",
    "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu",
    "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
    "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam",
    "Yemen", "Zambia", "Zimbabwe",
]


# ===========================
# Google Sheets logging
# ===========================
SHEET_KEY = "1lXyGAJm5MoO_NDhdBbI6u_o0C7vCLNGfI77MPipw8c8"

@st.cache_resource
def get_gsheet_worksheet():
    """Authorize with Google using service-account JSON stored in an env var.
    Returns the first worksheet, or None if logging is not available."""
    service_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not service_json:
        st.warning("Logging disabled: GOOGLE_SERVICE_ACCOUNT_JSON env var is not set.")
        return None

    # Parse JSON
    try:
        service_info = json.loads(service_json)
    except Exception as e:
        st.error("Logging disabled: could not parse GOOGLE_SERVICE_ACCOUNT_JSON.")
        st.exception(e)
        return None

    # Show which service account we are using (for debugging)
    sa_email = service_info.get("client_email", "UNKNOWN")

    # Build credentials and open sheet
    try:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(service_info, scopes=scopes)
        client = gspread.authorize(creds)
        sh = client.open_by_key(SHEET_KEY)
        return sh.sheet1
    except Exception as e:
        st.error("Logging disabled: failed to connect to Google Sheets.")
        st.exception(e)  # shows full traceback + API error JSON
        return None



def log_individual_prediction(
    mode,  # "real_data" or "exploration"
    bmi, age, waist, activity_label, smoker_label,
    sbp, dbp, hr, income_ratio, education_label,
    race_label, gender_label,
    p_diab, p_ckd, p_cvd,
    country="Prefer not to say",  # NEW: country of residence
):
    """
    Append a single anonymized row to the Google Sheet.

    All inputs are stored as coarse bins; only the model-predicted risks
    are stored as exact percentages. No timestamp or direct identifiers.

    Sheet header should be:

    mode,age_bin,bmi_bin,waist_bin,activity,smoker,
    sbp_bin,dbp_bin,hr_bin,income_ratio_bin,
    education,race,gender,
    diab_risk_pct,ckd_risk_pct,cvd_risk_pct,country
    """
    try:
        ws = get_gsheet_worksheet()
        if ws is None:
            # Logging not available; a warning/error was already shown by helper.
            return

        row = [
            mode,                        # mode: "real_data" or "exploration"
            bin_age(age),                # age_bin
            bin_bmi(bmi),                # bmi_bin
            bin_waist(waist),            # waist_bin
            activity_label,              # activity (Low/Moderate/High)
            smoker_label,                # smoker (Yes/No)
            bin_sbp(sbp),                # sbp_bin
            bin_dbp(dbp),                # dbp_bin
            bin_hr(hr),                  # hr_bin
            bin_income_ratio(income_ratio),  # income_ratio_bin
            education_label,             # education (NHANES categories)
            race_label,                  # race/ethnicity (4 cats)
            gender_label,                # gender
            float(p_diab) * 100.0,       # diab_risk_pct
            float(p_ckd) * 100.0,        # ckd_risk_pct
            float(p_cvd) * 100.0,        # cvd_risk_pct
            country,                     # country of residence (NEW)
        ]

        ws.append_row(row, value_input_option="USER_ENTERED")

    except Exception as e:
        st.error("Logging error: could not append row to Google Sheet.")
        st.write("Append error:", repr(e))


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

_CONSENT_TEXT = (
    "By using this app, you agree to the collection of anonymous, non-identifiable "
    "usage data (binned inputs and model-estimated risk outputs) for research purposes. "
    "No personal identifiers are collected."
)


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

# model trained with numeric "smoking"; we use 0/1 here
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
# Page navigation (URL-based)
# ===========================
# NOTE: Streamlit reserves "embed"/"embed_options" and strips them from st.query_params.
# We use "embedded=true" as our custom Python-readable embed flag instead.
# In Squarespace iframes use: ?embed=true&embedded=true
#   ?embed=true   → Streamlit hides its own chrome (toolbar/header/footer) natively
#   ?embedded=true → our Python code hides consent text and nav label
_embed_mode = st.query_params.get("embedded", "false").lower() == "true"

_PAGES = {
    "CVD/CKD/Diabetes Screener": "cvd-screener",
    "Prostate Cancer Screener": "prostate-screener",
    "Individual Sensitivity Analysis": "sensitivity",
    "Researcher Tools": "researcher-tools",
    "Global Validation Lab": "global-validation",
}
_page_names = list(_PAGES.keys())
_slugs = list(_PAGES.values())

_current_slug = st.query_params.get("page", "cvd-screener")
if _current_slug not in _slugs:
    _current_slug = "cvd-screener"
_current_idx = _slugs.index(_current_slug)

if _embed_mode:
    st.markdown("""
        <style>
        #MainMenu {visibility: hidden !important;}
        [data-testid="stHeader"] {display: none !important;}
        [data-testid="stToolbar"] {display: none !important;}
        [data-testid="stDecoration"] {display: none !important;}
        footer {display: none !important;}
        .stDeployButton {display: none !important;}
        </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    if not _embed_mode:
        st.markdown("### Navigation")
    _selected_name = st.radio(
        "", _page_names, index=_current_idx, label_visibility="collapsed"
    )
    _new_slug = _PAGES[_selected_name]
    if _new_slug != _current_slug:
        st.query_params["page"] = _new_slug
        if _embed_mode:
            st.query_params["embedded"] = "true"
        st.rerun()

page = _current_slug

if not _embed_mode:
    st.write(_CONSENT_TEXT)


# ============================================================
#                  CVD/CKD/DIABETES SCREENER
# ============================================================
if page == "cvd-screener":
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

    # ---------------- Country of residence ----------------
    country = st.selectbox(
        "Country of residence",
        COUNTRY_LIST,
        index=0,
        help=(
            "Select your country of residence. This is recorded only as a broad "
            "geographic indicator for research purposes. Choose 'Prefer not to say' "
            "to skip."
        ),
    )

    # Persist inputs so the Sensitivity Analysis page can use them
    st.session_state["cvd_age"] = age
    st.session_state["cvd_bmi"] = bmi
    st.session_state["cvd_waist"] = waist
    st.session_state["cvd_activity"] = activity
    st.session_state["cvd_smoker"] = smoker
    st.session_state["cvd_sbp"] = sbp
    st.session_state["cvd_dbp"] = dbp
    st.session_state["cvd_hr"] = hr
    st.session_state["cvd_income_ratio"] = income_ratio
    st.session_state["cvd_education"] = education
    st.session_state["cvd_race"] = race
    st.session_state["cvd_gender"] = gender

    # ---------------- Mode: real data vs exploring (forced choice) ----------------
    mode_choice = st.radio(
        "Are you entering your own real information, or just exploring the tool?",
        (
            "I'm entering my real information today",
            "I'm just exploring / using example values",
        ),
        index=None,   # <- forces no default selection
        help="This only affects how your data are logged for research; it does not change your risk estimates."
    )

    # Convert selection into mode variable
    if mode_choice is None:
        mode = None
    elif "real" in mode_choice:
        mode = "real_data"
    else:
        mode = "exploration"

    # ---------------- Predict ----------------
    if st.button("Estimate risks", key="calc_button"):
        # Force them to choose a mode first
        if mode is None:
            st.warning(
                "Please select whether you're using real information or just exploring before continuing."
            )
            st.stop()

        # 1) Build features
        X = build_feature_df(
            bmi=bmi, age=age, waist=waist, activity_label=activity,
            smoker_label=smoker, sbp=sbp, dbp=dbp, hr=hr,
            income_ratio=income_ratio, education_label=education,
            race_label=race, gender_label=gender
        )

        # 2) Predict
        X_model = prepare_for_model(X)
        p_diab, p_ckd, p_cvd = predict_three(X_model)

        # 3) Log (with mode and country)
        log_individual_prediction(
            mode=mode,
            bmi=bmi,
            age=age,
            waist=waist,
            activity_label=activity,
            smoker_label=smoker,
            sbp=sbp,
            dbp=dbp,
            hr=hr,
            income_ratio=income_ratio,
            education_label=education,
            race_label=race,
            gender_label=gender,
            p_diab=float(p_diab[0]),
            p_ckd=float(p_ckd[0]),
            p_cvd=float(p_cvd[0]),
            country=country,
        )

        # 4) Display
        r1, r2, r3 = st.columns(3)
        r1.metric("Diabetes risk", f"{p_diab[0]*100:.1f}%")
        r2.metric("CKD risk", f"{p_ckd[0]*100:.1f}%")
        r3.metric("CVD risk", f"{p_cvd[0]*100:.1f}%")

        st.caption("These estimates are based solely on non-dietary predictors.")


# ============================================================
#                  PROSTATE CANCER SCREENER
# ============================================================
elif page == "prostate-screener":
    st.title("Prostate Cancer Screener")

    st.subheader("Enter your information")

    psa_val = st.number_input("PSA level (ng/mL)", min_value=0.01, max_value=100.0, value=2.0, step=0.1)
    age_pca = st.number_input("Age (years)", min_value=18, max_value=100, value=60, key="pca_age")
    african_american = st.selectbox("Do you identify as African American / Black?", ["No", "Yes"], key="pca_aa")

    aa_flag = 1 if african_american == "Yes" else 0

    if st.button("Estimate prostate cancer risk", key="pca_button"):
        log2_psa = np.log2(psa_val)

        if psa_val >= 2:
            S = (-6.84249970
                 + 0.70043815 * log2_psa
                 + 0.04574460 * age_pca
                 + 1.01699029 * aa_flag)
        else:
            S = (-2.81814489
                 + 0.24044370 * log2_psa
                 + 0.01370219 * age_pca
                 + 0.12000825 * aa_flag)

        p_no_cancer = 1.0 / (1.0 + np.exp(S))
        p_cancer = 1.0 - p_no_cancer
        p_cancer_pct = p_cancer * 100.0

        st.metric("Estimated probability of prostate cancer", f"{p_cancer_pct:.1f}%")

        if psa_val >= 2:
            st.warning(
                f"You would likely be recommended for a biopsy using standard tests, "
                f"yet your approximate risk of having prostate cancer is only {p_cancer_pct:.1f}%. "
                f"Help fund glycoscore to prevent these unnecessary biopsies: "
                f"750,000 of them per year in the US alone."
            )
        else:
            st.info(
                "You would not be recommended for a biopsy using standard tests, "
                "but millions of adults receive unnecessary biopsies each year because of "
                "these tests. Help fund glycoscore to prevent these unnecessary biopsies: "
                "750,000 of them per year in the US alone."
            )



    st.caption(
        "Research prototype. Not for clinical use. "
        "This estimate is based on many assumptions and should not replace "
        "medical advice, screening, or biopsy."
    )


# ============================================================
#               INDIVIDUAL SENSITIVITY ANALYSIS
# ============================================================
elif page == "sensitivity":
    st.title("Individual sensitivity analysis")
    st.caption(
        "These plots use the **same profile** you entered in the "
        "'CVD/CKD/Diabetes Screener' page as the baseline individual."
    )

    # Read from session state (populated by CVD screener); fall back to sensible defaults
    age = st.session_state.get("cvd_age", 45)
    bmi = st.session_state.get("cvd_bmi", 24.7)
    waist = st.session_state.get("cvd_waist", 95.0)
    activity = st.session_state.get("cvd_activity", "Moderate")
    smoker = st.session_state.get("cvd_smoker", "No")
    sbp = st.session_state.get("cvd_sbp", 120)
    dbp = st.session_state.get("cvd_dbp", 75)
    hr = st.session_state.get("cvd_hr", 70)
    income_ratio = st.session_state.get("cvd_income_ratio", 2.0)
    education = st.session_state.get("cvd_education", "High school graduate/GED or equivalent")
    race = st.session_state.get("cvd_race", "Non-Hispanic White")
    gender = st.session_state.get("cvd_gender", "Male")

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
#                 RESEARCHER TOOLS (POPULATION-BASED)
# ============================================================
elif page == "researcher-tools":
    # --- Terms of use gate for Researcher tools tab ---
    if "researcher_agreed" not in st.session_state:
        st.session_state["researcher_agreed"] = False

    if not st.session_state["researcher_agreed"]:
        st.markdown("### Terms of use for Researcher tools")
        st.markdown(
            """
        By using the **Researcher tools** in this app, you agree to the following:

        1. For exploratory **internal** use—such as lab discussions, coursework, or preliminary
           non-public analyses—you may use figures or tables generated by this tool, provided
           you cite:

           **Costa, AM and Iris, BD. Nature Scientific Reports. 2026.**

        2. For **any formal use** of this tool or its outputs in publications—including use in
           **manuscripts, preprints, external talks, or grant proposals**—you agree to
           **contact Arthur M. Costa (arthurcosta@uchicago.edu)** to discuss collaboration and
           **appropriate authorship**.

           The authors of Costa, AM and Iris, BD. Nature Scientific Reports. 2026. curated the
           underlying dataset, engineered derived variables, and developed the modelling
           and population-simulation framework used in this tool. Guidance on methodology
           can be found in the publication above, or by contacting Arthur directly.

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
            # Stop rendering anything else until they agree
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
            max_n = int(min(100000, nhanes_sim.shape[0]))
            pop_n = st.slider(
                "Population size",
                min_value=500,
                max_value=max_n,
                value=min(3000, max_n),
                step=500,
                key="pop_n",
            )
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
            available_idx = nhanes_sim.index.to_numpy()
            sampled_idx = rng.choice(available_idx, size=pop_n, replace=False)
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

            fig_change = apply_plotly_figure_editor(
                fig_change,
                key_prefix="overall_rel_change",
                default_title="Overall relative change in mean predicted risk",
                default_x="Disease",
                default_y="Relative change in mean predicted risk (%)",
            )

            st.plotly_chart(fig_change, use_container_width=True,
                            config=PLOTLY_DOWNLOAD_CONFIG)

        if "overall_df" in st.session_state:
            overall = st.session_state["overall_df"]

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
                title=f"Model-predicted mean risk by {pop_sens_label.split('(')[0].strip().lower()}",
                xaxis_title=pop_sens_label,  # e.g. "Age (years)"
                yaxis_title="Mean model-predicted probability (%)",
                margin=dict(l=60, r=20, t=80, b=60),
            )

            fig_pop_sens = style_plotly_pub(fig_pop_sens)

            fig_pop_sens = apply_plotly_figure_editor(
                fig_pop_sens,
                key_prefix="pop_1d_sensitivity",
                default_title=f"Model-predicted mean risk by {pop_sens_label.split('(')[0].strip().lower()}",
                default_x=pop_sens_label,
                default_y="Mean model-predicted probability (%)",
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

            # Re-apply single shared axis titles so they don't repeat on all 3 panels
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



# ============================================================
#                  GLOBAL VALIDATION LAB
# ============================================================
elif page == "global-validation":
    import os as _os
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, brier_score_loss, roc_curve,
        precision_recall_curve,
    )
    try:
        from xgboost import XGBClassifier
        _HAS_XGB = True
    except Exception:
        _HAS_XGB = False

    st.title("Global Validation Lab")
    st.caption(
        "Validate any non-laboratory (or laboratory) screening model across up to 90 countries — "
        "WHO STEPS + NHANES + KNHANES, ~365,000 adults. Build a model here or upload your own, "
        "choose where to train and where to test, and get per-country discrimination, calibration, "
        "and operating-point metrics with downloadable tables and figures."
    )

    # -----------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------
    @st.cache_data(show_spinner="Loading harmonized global dataset…")
    def _gv_load():
        here = _os.path.dirname(_os.path.abspath(__file__))
        cands = []
        for base in [here, _os.getcwd(), _os.path.join(here, "data")]:
            cands += [_os.path.join(base, "harmonized_global.parquet"),
                      _os.path.join(base, "harmonized_global.xlsx")]
        for p in cands:
            if _os.path.exists(p):
                df = pd.read_parquet(p) if p.endswith("parquet") else pd.read_excel(p, sheet_name="data")
                return df, p
        return None, None

    gv_data, gv_path = _gv_load()
    if gv_data is None:
        st.error(
            "Harmonized dataset not found. Place **harmonized_global.parquet** (or "
            "**harmonized_global.xlsx**) next to app.py."
        )
        st.stop()

    # -----------------------------------------------------------------
    # Variable metadata
    # -----------------------------------------------------------------
    GV_CAT = {"sex", "race", "education", "region", "income_group", "urban_rural", "subpopulation"}
    GV_GROUPS = {
        "Demographics & context": ["age", "sex", "race", "education", "region",
                                   "income_group", "year", "urban_rural"],
        "Anthropometrics": ["height_cm", "weight_kg", "bmi", "waist_cm", "hip_cm", "waist_hip_ratio"],
        "Vitals": ["systolic", "diastolic", "heart_rate", "pulse_pressure"],
        "Tobacco": ["tobacco_current", "tobacco_daily", "smokeless_tobacco"],
        "Alcohol": ["alcohol_ever", "alcohol_current"],
        "Diet": ["fruit_days", "fruit_servings", "veg_days", "veg_servings", "adds_salt_often"],
        "Physical activity": ["activity_level", "activity_vigorous_work", "activity_moderate_work",
                              "active_transport", "recreation_vigorous", "recreation_moderate"],
        "Laboratory": ["fasting_glucose", "hba1c", "total_cholesterol", "ldl", "triglycerides",
                       "creatinine", "egfr", "crp"],
        "History & medications": ["told_high_bp", "on_bp_meds", "told_high_glucose",
                                  "on_glucose_meds", "family_history_cvd", "prior_mi_stroke"],
    }
    GV_LABELS = {
        "age": "Age (years)", "sex": "Sex", "race": "Race/ethnicity (NHANES)",
        "education": "Education level", "region": "WHO region", "income_group": "World Bank income group",
        "year": "Survey year", "urban_rural": "Urban/Rural", "subpopulation": "Subpopulation",
        "height_cm": "Height (cm)", "weight_kg": "Weight (kg)", "bmi": "BMI (kg/m²)",
        "waist_cm": "Waist circumference (cm)", "hip_cm": "Hip circumference (cm)",
        "waist_hip_ratio": "Waist–hip ratio", "systolic": "Systolic BP (mmHg)",
        "diastolic": "Diastolic BP (mmHg)", "heart_rate": "Heart rate (bpm)",
        "pulse_pressure": "Pulse pressure (mmHg)", "tobacco_current": "Current tobacco use (0/1)",
        "tobacco_daily": "Daily tobacco use (0/1)", "smokeless_tobacco": "Smokeless tobacco (0/1)",
        "alcohol_ever": "Ever consumed alcohol (0/1)", "alcohol_current": "Current alcohol use (0/1)",
        "fruit_days": "Fruit intake (days/week)", "fruit_servings": "Fruit (servings/day)",
        "veg_days": "Vegetable intake (days/week)", "veg_servings": "Vegetables (servings/day)",
        "adds_salt_often": "Often adds salt (0/1)", "activity_level": "Physical activity level (0–2)",
        "activity_vigorous_work": "Vigorous work activity (0/1)",
        "activity_moderate_work": "Moderate work activity (0/1)",
        "active_transport": "Active transport (walk/bike, 0/1)",
        "recreation_vigorous": "Vigorous recreation (0/1)", "recreation_moderate": "Moderate recreation (0/1)",
        "fasting_glucose": "Fasting glucose (mg/dL)", "hba1c": "HbA1c (%)",
        "total_cholesterol": "Total cholesterol (mg/dL)", "ldl": "LDL cholesterol (mg/dL)",
        "triglycerides": "Triglycerides (mg/dL)", "creatinine": "Creatinine (mg/dL)",
        "egfr": "eGFR (mL/min/1.73m²)", "crp": "C-reactive protein (mg/L)",
        "told_high_bp": "Told has high BP (0/1)", "on_bp_meds": "On BP medication (0/1)",
        "told_high_glucose": "Told has high glucose (0/1)", "on_glucose_meds": "On glucose medication (0/1)",
        "family_history_cvd": "Family history of CVD (0/1)", "prior_mi_stroke": "Prior MI or stroke (0/1)",
    }
    GV_OUTCOMES = {
        "dx_diabetes": "Diabetes (prevalent)", "dx_cvd": "Cardiovascular disease (MI/stroke)",
        "dx_hypertension": "Hypertension (measured ≥140/90)", "dx_ckd": "Chronic kidney disease (eGFR<60)",
        "dx_obesity": "Obesity (BMI ≥ 30)",
    }
    GV_PRED_CANDIDATES = [c for g in GV_GROUPS.values() for c in g]
    GV_NUMERIC_FOR_CUTOFF = ["fasting_glucose", "hba1c", "total_cholesterol", "ldl", "triglycerides",
                             "creatinine", "egfr", "crp", "systolic", "diastolic", "pulse_pressure",
                             "bmi", "waist_cm", "hip_cm", "waist_hip_ratio", "age", "heart_rate"]

    def _gv_label(c):
        return GV_LABELS.get(c, GV_OUTCOMES.get(c, c))

    # -----------------------------------------------------------------
    # Sidebar-style configuration (main pane)
    # -----------------------------------------------------------------
    with st.expander("① Dataset overview", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Participants", f"{len(gv_data):,}")
        c2.metric("Countries", f"{gv_data['iso3'].nunique()}")
        c3.metric("WHO regions", f"{gv_data['region'].nunique()}")
        c4.metric("Data sources", f"{gv_data['source'].nunique()}")
        src_tbl = (gv_data.groupby("source")
                   .agg(participants=("age", "size"), countries=("iso3", "nunique"))
                   .reset_index())
        st.dataframe(src_tbl, use_container_width=True, hide_index=True)
        st.caption(f"Loaded from `{_os.path.basename(gv_path)}`.")

    st.markdown("### ② Define the model")
    gv_mode = st.radio(
        "How do you want to supply the model?",
        ["Build a model here", "Upload a .joblib model"],
        horizontal=True,
    )

    # ----- outcome -----
    st.markdown("#### Outcome (dependent variable)")
    oc_col1, oc_col2 = st.columns([2, 3])
    with oc_col1:
        gv_out_kind = st.radio("Outcome type", ["Pre-built binary outcome", "Numeric variable + cut-off"],
                               key="gv_out_kind")
    gv_cut_dir, gv_cut_val = None, None
    if gv_out_kind == "Pre-built binary outcome":
        gv_outcome = st.selectbox("Outcome", list(GV_OUTCOMES.keys()),
                                  format_func=lambda c: GV_OUTCOMES[c])
    else:
        with oc_col2:
            gv_outcome = st.selectbox("Numeric variable", GV_NUMERIC_FOR_CUTOFF,
                                      format_func=_gv_label)
            _vals = pd.to_numeric(gv_data[gv_outcome], errors="coerce")
            d1, d2 = st.columns(2)
            gv_cut_dir = d1.selectbox("Positive when value is", ["≥ cut-off", "≤ cut-off"])
            default_cut = float(np.nanmedian(_vals))
            gv_cut_val = d2.number_input("Cut-off", value=round(default_cut, 1))
            st.caption(f"Positive = {gv_outcome} {'≥' if gv_cut_dir.startswith('≥') else '≤'} {gv_cut_val}. "
                       f"(observed range ≈ {np.nanmin(_vals):.0f}–{np.nanmax(_vals):.0f})")

    def _gv_make_outcome(df):
        if gv_out_kind == "Pre-built binary outcome":
            y = pd.to_numeric(df[gv_outcome], errors="coerce")
            return y.where(y.isin([0, 1]))
        v = pd.to_numeric(df[gv_outcome], errors="coerce")
        y = pd.Series(np.nan, index=df.index)
        y[v.notna()] = 0.0
        if gv_cut_dir.startswith("≥"):
            y[v >= gv_cut_val] = 1.0
        else:
            y[v <= gv_cut_val] = 1.0
        return y

    # ----- predictors -----
    st.markdown("#### Predictors (independent variables)")
    if gv_mode == "Build a model here":
        default_preds = ["age", "sex", "bmi", "waist_cm", "systolic"]
        gv_preds = st.multiselect(
            "Select predictors", GV_PRED_CANDIDATES,
            default=[p for p in default_preds if p != gv_outcome],
            format_func=_gv_label,
        )
        gv_preds = [p for p in gv_preds if p != gv_outcome]
        gv_model_type = st.selectbox(
            "Model type",
            (["Logistic regression", "Random forest"] + (["XGBoost"] if _HAS_XGB else [])),
        )
    else:
        st.info("Upload a fitted scikit-learn / XGBoost model saved with joblib. "
                "Then map its input features, in order, to harmonized columns.")
        gv_upload = st.file_uploader("Upload .joblib model", type=["joblib", "pkl"])
        gv_preds = st.multiselect(
            "Harmonized columns feeding the model (IN THE ORDER THE MODEL EXPECTS)",
            GV_PRED_CANDIDATES, default=[], format_func=_gv_label,
        )
        gv_model_type = "Uploaded"

    # -----------------------------------------------------------------
    # Availability-aware country eligibility
    # -----------------------------------------------------------------
    MIN_TEST_N = st.slider("Minimum complete-case participants per test country", 30, 500, 100, 10)
    MIN_POS = 10

    @st.cache_data(show_spinner=False)
    def _gv_country_eligibility(preds, outcome_key, out_kind, cut_dir, cut_val, min_n, min_pos):
        needed = [c for c in preds if c in gv_data.columns]
        y = _gv_make_outcome(gv_data)
        base = gv_data[needed].notna().all(axis=1) if needed else pd.Series(True, index=gv_data.index)
        ok = base & y.isin([0, 1])
        rows = []
        for iso, g in gv_data.assign(_ok=ok, _y=y).groupby("iso3"):
            gg = g[g["_ok"]]
            npos = int((gg["_y"] == 1).sum()); nneg = int((gg["_y"] == 0).sum())
            rows.append(dict(iso3=iso, country=g["country"].iloc[0], region=g["region"].iloc[0],
                             source=g["source"].iloc[0], n=len(gg), n_pos=npos, n_neg=nneg,
                             eligible=(len(gg) >= min_n and npos >= min_pos and nneg >= min_pos)))
        return pd.DataFrame(rows).sort_values(["eligible", "n"], ascending=[False, False])

    if (gv_mode == "Build a model here" and gv_preds) or (gv_mode == "Upload a .joblib model" and gv_preds):
        elig = _gv_country_eligibility(tuple(gv_preds), gv_outcome, gv_out_kind,
                                       gv_cut_dir, gv_cut_val, MIN_TEST_N, MIN_POS)
    else:
        elig = pd.DataFrame(columns=["iso3", "country", "region", "source", "n", "n_pos", "n_neg", "eligible"])

    elig_countries = elig[elig.eligible]["country"].tolist()
    n_excluded = int((~elig.eligible).sum()) if len(elig) else 0

    st.markdown("### ③ Choose where to train and where to test")
    st.caption(
        f"With your current variables, **{len(elig_countries)} countries** have enough complete-case "
        f"data (≥{MIN_TEST_N} participants and ≥{MIN_POS} positives & negatives). "
        f"{n_excluded} countries are hidden because the selected variables are missing there."
    )

    presets = {
        "Train USA (NHANES) → test all other eligible countries": "usa_world",
        "Train South Korea (KNHANES) → test all other eligible countries": "kor_world",
        "Train on ALL eligible → test on ALL eligible (pooled)": "pooled",
        "Custom selection": "custom",
    }
    gv_preset = st.selectbox("Preset", list(presets.keys()))
    preset = presets[gv_preset]

    def _sel_from_scope(scope):
        return elig[elig.eligible & elig[scope[0]].eq(scope[1])]["country"].tolist()

    if preset == "custom":
        cc1, cc2 = st.columns(2)
        with cc1:
            train_countries = st.multiselect("Train countries", elig_countries,
                                             default=[c for c in ["United States"] if c in elig_countries])
        with cc2:
            test_countries = st.multiselect("Test countries", elig_countries,
                                            default=[c for c in elig_countries if c not in train_countries][:40])
    elif preset == "usa_world":
        train_countries = [c for c in ["United States"] if c in elig_countries]
        test_countries = [c for c in elig_countries if c != "United States"]
    elif preset == "kor_world":
        train_countries = [c for c in ["South Korea"] if c in elig_countries]
        test_countries = [c for c in elig_countries if c != "South Korea"]
    else:  # pooled
        train_countries = elig_countries
        test_countries = elig_countries

    if gv_mode == "Upload a .joblib model":
        train_countries = []  # no training needed

    with st.expander("Show country availability table", expanded=False):
        st.dataframe(elig.rename(columns={"n": "complete_cases"}), use_container_width=True, hide_index=True)

    # threshold for operating-point metrics
    gv_thr = st.slider("Decision threshold for sensitivity / specificity / PPV", 0.01, 0.99, 0.50, 0.01)

    run_gv = st.button("🚀 Run global validation", type="primary", use_container_width=True)

    # -----------------------------------------------------------------
    # Metric helpers
    # -----------------------------------------------------------------
    def _gv_calib(y, p):
        eps = 1e-6
        pc = np.clip(p, eps, 1 - eps)
        logit = np.log(pc / (1 - pc))
        try:
            lr = LogisticRegression(max_iter=1000).fit(logit.reshape(-1, 1), y)
            return float(lr.coef_[0][0]), float(lr.intercept_[0])
        except Exception:
            return np.nan, np.nan

    def _gv_metrics(y, p, thr):
        y = np.asarray(y).astype(int); p = np.asarray(p, float)
        out = dict(n=len(y), n_pos=int(y.sum()), prevalence=float(y.mean()))
        if y.sum() == 0 or y.sum() == len(y):
            out.update(dict(AUROC=np.nan, PR_AUC=np.nan, Brier=np.nan, calib_slope=np.nan,
                            calib_intercept=np.nan, sensitivity=np.nan, specificity=np.nan,
                            PPV=np.nan, NPV=np.nan))
            return out
        out["AUROC"] = float(roc_auc_score(y, p))
        out["PR_AUC"] = float(average_precision_score(y, p))
        out["Brier"] = float(brier_score_loss(y, p))
        s, i = _gv_calib(y, p); out["calib_slope"], out["calib_intercept"] = s, i
        yhat = (p >= thr).astype(int)
        tp = int(((yhat == 1) & (y == 1)).sum()); fp = int(((yhat == 1) & (y == 0)).sum())
        tn = int(((yhat == 0) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
        out["sensitivity"] = tp / (tp + fn) if (tp + fn) else np.nan
        out["specificity"] = tn / (tn + fp) if (tn + fp) else np.nan
        out["PPV"] = tp / (tp + fp) if (tp + fp) else np.nan
        out["NPV"] = tn / (tn + fn) if (tn + fn) else np.nan
        return out

    def _gv_build_pipeline(preds, model_type):
        num = [c for c in preds if c not in GV_CAT]
        cat = [c for c in preds if c in GV_CAT]
        if model_type == "Logistic regression":
            num_tf = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
            clf = LogisticRegression(max_iter=2000)
        elif model_type == "Random forest":
            num_tf = Pipeline([("imp", SimpleImputer(strategy="median"))])
            clf = RandomForestClassifier(n_estimators=400, max_depth=8, n_jobs=-1, random_state=42)
        else:  # XGBoost
            num_tf = Pipeline([("imp", SimpleImputer(strategy="median"))])
            clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                subsample=0.85, colsample_bytree=0.85, eval_metric="logloss",
                                n_jobs=-1, random_state=42, tree_method="hist")
        pre = ColumnTransformer([
            ("num", num_tf, num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ])
        return Pipeline([("pre", pre), ("clf", clf)])

    # -----------------------------------------------------------------
    # RUN
    # -----------------------------------------------------------------
    if run_gv:
        if not gv_preds:
            st.error("Select at least one predictor."); st.stop()
        if not test_countries:
            st.error("No eligible test countries with the current variables/threshold."); st.stop()

        y_all = _gv_make_outcome(gv_data)
        work = gv_data.copy(); work["_y"] = y_all
        needed = gv_preds
        complete = work[needed].notna().all(axis=1) & work["_y"].isin([0, 1])
        work = work[complete]

        # ---- model ----
        model, train_note = None, ""
        if gv_mode == "Upload a .joblib model":
            if gv_upload is None:
                st.error("Upload a .joblib model first."); st.stop()
            try:
                model = joblib.load(gv_upload)
            except Exception as e:
                st.error(f"Could not load model: {e}"); st.stop()
            train_note = "Uploaded model (no training performed)."
        else:
            tr = work[work.country.isin(train_countries)]
            if tr["_y"].nunique() < 2 or len(tr) < 50:
                st.error("Training set has too few samples or only one outcome class."); st.stop()
            with st.spinner(f"Training {gv_model_type} on {len(tr):,} participants…"):
                model = _gv_build_pipeline(gv_preds, gv_model_type)
                model.fit(tr[gv_preds], tr["_y"].astype(int))
            train_note = f"{gv_model_type} trained on {len(tr):,} participants from {len(train_countries)} country(ies)."

        def _predict(df):
            X = df[gv_preds]
            try:
                return model.predict_proba(X)[:, 1]
            except Exception:
                # uploaded model may expect raw numpy / different column names
                return np.asarray(model.predict_proba(X.values))[:, 1]

        # ---- per-country evaluation ----
        te = work[work.country.isin(test_countries)]
        results = []
        for (iso, country, region, source), g in te.groupby(["iso3", "country", "region", "source"]):
            try:
                p = _predict(g)
            except Exception as e:
                continue
            m = _gv_metrics(g["_y"].values, p, gv_thr)
            m.update(dict(iso3=iso, country=country, region=region, source=source))
            results.append(m)
        res = pd.DataFrame(results)
        if res.empty:
            st.error("No test countries could be scored (prediction failed or no valid data)."); st.stop()

        # pooled prediction for global curves
        p_pool = _predict(te)
        y_pool = te["_y"].values
        pooled = _gv_metrics(y_pool, p_pool, gv_thr)

        st.session_state["gv_out"] = dict(
            res=res, pooled=pooled, y_pool=y_pool, p_pool=p_pool,
            te_assembled=te[["source", "country", "iso3", "region"] + gv_preds + ["_y"]].rename(columns={"_y": "outcome"}),
            train_note=train_note, preds=gv_preds, outcome=gv_outcome, thr=gv_thr,
            model_type=gv_model_type, region_series=te["region"].values,
        )

    # -----------------------------------------------------------------
    # RESULTS
    # -----------------------------------------------------------------
    if "gv_out" in st.session_state:
        o = st.session_state["gv_out"]
        res, pooled = o["res"], o["pooled"]
        st.success(o["train_note"])
        st.markdown("### ④ Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test countries", f"{res['iso3'].nunique()}")
        m2.metric("Median country AUROC", f"{res['AUROC'].median():.3f}")
        m3.metric("Pooled AUROC", f"{pooled['AUROC']:.3f}" if pooled['AUROC'] == pooled['AUROC'] else "—")
        m4.metric("Countries AUROC ≥ 0.70", f"{int((res['AUROC'] >= 0.70).sum())}/{res['AUROC'].notna().sum()}")

        # ---- world map ----
        st.markdown("#### Performance map")
        map_metric = st.selectbox("Colour countries by",
                                  ["AUROC", "PR_AUC", "Brier", "calib_slope", "sensitivity", "specificity", "prevalence"],
                                  key="gv_map_metric")
        fig_map = go.Figure(go.Choropleth(
            locations=res["iso3"], z=res[map_metric], locationmode="ISO-3",
            colorscale="RdYlGn" if map_metric in ("AUROC", "PR_AUC", "sensitivity", "specificity") else "Viridis",
            marker_line_color="white", marker_line_width=0.4,
            colorbar_title=map_metric, zmid=(0.7 if map_metric == "AUROC" else None),
            text=res["country"], hovertemplate="%{text}<br>" + map_metric + "=%{z:.3f}<extra></extra>",
        ))
        fig_map.update_layout(geo=dict(showframe=False, showcoastlines=False, projection_type="natural earth"),
                              margin=dict(l=0, r=0, t=10, b=0), height=460)
        st.plotly_chart(fig_map, use_container_width=True, config=PLOTLY_DOWNLOAD_CONFIG)

        # ---- ROC + calibration ----
        cc1, cc2 = st.columns(2)
        with cc1:
            st.markdown("#### ROC (pooled test set)")
            if pooled["AUROC"] == pooled["AUROC"]:
                fpr, tpr, _ = roc_curve(o["y_pool"], o["p_pool"])
                figroc = go.Figure()
                figroc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                            name=f"AUROC={pooled['AUROC']:.3f}", line=dict(color=COLOR_DIAB, width=3)))
                figroc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                            line=dict(color="grey", dash="dash"), showlegend=False))
                figroc.update_layout(xaxis_title="1 − specificity", yaxis_title="sensitivity",
                                     height=380, margin=dict(l=50, r=10, t=10, b=40))
                st.plotly_chart(figroc, use_container_width=True, config=PLOTLY_DOWNLOAD_CONFIG)
        with cc2:
            st.markdown("#### Calibration (pooled test set)")
            dfc = pd.DataFrame({"y": o["y_pool"], "p": o["p_pool"]})
            dfc["bin"] = pd.qcut(dfc["p"].rank(method="first"), 10, labels=False)
            cal = dfc.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean")).reset_index()
            figcal = go.Figure()
            figcal.add_trace(go.Scatter(x=cal["pred"], y=cal["obs"], mode="lines+markers",
                                        line=dict(color=COLOR_CVD, width=3), name="observed"))
            figcal.add_trace(go.Scatter(x=[0, cal["pred"].max()], y=[0, cal["pred"].max()], mode="lines",
                                        line=dict(color="grey", dash="dash"), showlegend=False))
            figcal.update_layout(xaxis_title="mean predicted risk", yaxis_title="observed frequency",
                                 height=380, margin=dict(l=50, r=10, t=10, b=40))
            st.plotly_chart(figcal, use_container_width=True, config=PLOTLY_DOWNLOAD_CONFIG)

        # ---- metric distribution across countries ----
        st.markdown("#### Distribution of country-level performance")
        dist_metric = st.selectbox("Metric", ["AUROC", "PR_AUC", "Brier", "sensitivity", "specificity", "calib_slope"],
                                   key="gv_dist_metric")
        figbox = go.Figure()
        for reg, g in res.dropna(subset=[dist_metric]).groupby("region"):
            figbox.add_trace(go.Box(y=g[dist_metric], name=reg, boxpoints="all", jitter=0.4,
                                    pointpos=0, marker=dict(size=5), text=g["country"]))
        figbox.update_layout(yaxis_title=dist_metric, height=420, showlegend=False,
                             margin=dict(l=50, r=10, t=10, b=40))
        st.plotly_chart(figbox, use_container_width=True, config=PLOTLY_DOWNLOAD_CONFIG)

        # ---- tables + downloads ----
        st.markdown("#### Per-country metric table")
        show_cols = ["country", "iso3", "region", "source", "n", "n_pos", "prevalence",
                     "AUROC", "PR_AUC", "Brier", "calib_slope", "calib_intercept",
                     "sensitivity", "specificity", "PPV", "NPV"]
        res_show = res[show_cols].sort_values("AUROC", ascending=False)
        st.dataframe(res_show.round(4), use_container_width=True, hide_index=True)

        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ Per-country metrics (CSV)",
                           res_show.round(5).to_csv(index=False).encode(),
                           "global_validation_metrics.csv", "text/csv", use_container_width=True)
        # cohort summary stats
        summ = (o["te_assembled"].drop(columns=["iso3"])
                .describe(include="all").transpose().reset_index().rename(columns={"index": "variable"}))
        d2.download_button("⬇ Cohort summary stats (CSV)",
                           summ.to_csv(index=False).encode(),
                           "global_validation_summary.csv", "text/csv", use_container_width=True)
        d3.download_button("⬇ Full analysis table (CSV)",
                           o["te_assembled"].to_csv(index=False).encode(),
                           "global_validation_data.csv", "text/csv", use_container_width=True)

        with st.expander("Cohort summary statistics", expanded=False):
            st.dataframe(summ, use_container_width=True, hide_index=True)

        st.caption(
            "Notes — AUROC and PR-AUC measure discrimination; calibration slope/intercept and Brier "
            "measure calibration; sensitivity/specificity/PPV/NPV are computed at your chosen decision "
            "threshold. Prevalence-dependent metrics (PR-AUC, PPV, calibration-in-the-large) are expected "
            "to vary more across countries than AUROC. Research prototype — not for clinical use."
        )
