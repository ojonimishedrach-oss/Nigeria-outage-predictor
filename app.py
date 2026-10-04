import joblib, pandas as pd, streamlit as st

pkg = joblib.load("model.pkl")
st.title("Nigeria Grid Collapse Predictor")

rain = st.slider("Rainfall today (mm)", 0.0, 80.0, 5.0)
rain3 = st.slider("Rainfall, last 3 days (mm)", 0.0, 200.0, 15.0)
rain7 = st.slider("Rainfall, last 7 days (mm)", 0.0, 400.0, 35.0)
temp = st.slider("Temperature (C)", 15.0, 40.0, 27.0)
temp7 = st.slider("Avg temperature, last 7 days (C)", 15.0, 40.0, 27.0)
month = st.selectbox("Month", list(range(1, 13)))
weekend = st.checkbox("Weekend")
since = st.number_input("Days since last collapse", 0, 999, 30)
c90 = st.number_input("Collapses in last 90 days", 0, 20, 1)

if st.button("Predict"):
    row = pd.DataFrame([{
        "rainfall": rain, "rain_3d": rain3, "rain_7d": rain7,
        "temp": temp, "temp_7d": temp7,
        "heavy_rain": int(rain > pkg["rain_q90"]),
        "harmattan_flag": int(month in [12, 1, 2]),
        "rainy": int(4 <= month <= 10),
        "month": month, "is_weekend": int(weekend),
        "days_since_collapse": since, "collapses_90d": c90,
    }])[pkg["features"]]
    p = pkg["model"].predict_proba(row)[0][1]
    st.metric("Collapse risk tomorrow", f"{p:.1%}")
    st.warning("High risk") if p > 0.5 else st.success("Lower risk")
