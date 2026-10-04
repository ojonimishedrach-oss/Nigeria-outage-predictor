import joblib, pandas as pd
from fastapi import FastAPI

app = FastAPI()
pkg = joblib.load("model.pkl")

@app.post("/predict")
def predict(data: dict):
    row = pd.DataFrame([data])[pkg["features"]]
    return {"collapse_risk": float(pkg["model"].predict_proba(row)[0][1])}
