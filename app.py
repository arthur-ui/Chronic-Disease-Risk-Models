if st.button("Estimate diabetes risk"):
    X_raw = pd.DataFrame([{
        "bmi": bmi,
        "AgeYears": age,
        "waist_circumference": waist,
        "activity_level": activity_map[activity],
        "smoking": smoke_map[smoker],
        "avg_systolic": sbp,
        "avg_diastolic": dbp,
        "avg_HR": hr,
        "FamIncome_to_poverty_ratio": income,
        "Education": None,  # you can add a widget later
        "Race": race_map[race],
        "Gender": gender_map[gender],
    }])

    proba = float(pipe.predict_proba(X_raw)[0, 1])
    st.metric("Estimated current diabetes risk", f"{proba*100:.1f}%")

    st.caption(
        "This estimate is based on non-dietary predictors only and is intended "
        "for research and educational purposes, not for diagnosis or treatment."
    )

st.markdown("---")
st.markdown("**Model details**")
st.markdown(
    "- Model: Bagged decision trees with preprocessing pipeline\n"
    "- Training data: NHANES 2011–2016, validated on 2017–2020\n"
    "- Inputs: anthropometrics, vital signs, and sociodemographic variables"
)
