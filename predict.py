"""Loads the trained model and predicts for one student. Used by app.py."""
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_model():
    bundle = joblib.load(ROOT / "models" / "model.joblib")
    return bundle["model"], bundle["features"]


def grade_category(marks: float) -> str:
    if marks >= 80:
        return "Excellent"
    if marks >= 65:
        return "Good"
    if marks >= 50:
        return "Average"
    return "At Risk"


def predict_student(answers: dict) -> dict:
    """answers: {feature_name: value}. Returns predicted marks + category."""
    model, features = load_model()
    row = pd.DataFrame([answers])[features]          # keep correct column order
    marks = float(model.predict(row)[0])
    marks = max(0.0, min(100.0, marks))
    return {"marks": round(marks, 1), "category": grade_category(marks)}


def load_feature_importance() -> pd.Series:
    path = ROOT / "outputs" / "feature_importance.json"
    return pd.Series(json.loads(path.read_text())).sort_values(ascending=False)
