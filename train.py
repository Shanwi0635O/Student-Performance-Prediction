"""Clean -> split -> train 3 models -> evaluate -> save best model + charts.
Run:  python src/train.py
"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TARGET = "Final_Marks"

# ---------- 1. Load & clean ----------
df = pd.read_csv("data/student_performance.csv")
print("Raw shape:", df.shape)
df = df.drop_duplicates()                       # remove duplicate rows
df = df.fillna(df.median(numeric_only=True))    # fill missing values with column median
print("Clean shape:", df.shape, "| missing left:", int(df.isnull().sum().sum()))

X = df.drop(columns=[TARGET])
y = df[TARGET]
feature_names = list(X.columns)

# ---------- 2. Train / test split (80 / 20) ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train:", X_train.shape, "Test:", X_test.shape)

# ---------- 3. Train several models ----------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results[name] = {
        "MAE": round(mean_absolute_error(y_test, pred), 2),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 2),
        "R2": round(r2_score(y_test, pred), 3),
    }
    print(f"{name:20s} {results[name]}")

# ---------- 4. Pick best by R2 and save ----------
best_name = max(results, key=lambda n: results[n]["R2"])
best_model = models[best_name]
print("\nBest model:", best_name)
joblib.dump({"model": best_model, "features": feature_names}, "models/model.joblib")

with open("outputs/metrics.json", "w") as f:
    json.dump({"best_model": best_name, "results": results}, f, indent=2)

# ---------- 5. Visualisations ----------
# (a) Model comparison
res_df = pd.DataFrame(results).T
ax = res_df[["MAE", "RMSE"]].plot(kind="bar", figsize=(7, 4), color=["#4C72B0", "#DD8452"])
ax.set_title("Model Comparison (lower is better)")
ax.set_ylabel("Marks error")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("outputs/04_model_comparison.png", dpi=150)
plt.close()

# (b) Actual vs predicted for best model
pred = best_model.predict(X_test)
plt.figure(figsize=(5.5, 5.5))
plt.scatter(y_test, pred, alpha=0.5)
lims = [0, 100]
plt.plot(lims, lims, "r--", label="Perfect prediction")
plt.xlabel("Actual Final Marks")
plt.ylabel("Predicted Final Marks")
plt.title(f"Actual vs Predicted ({best_name})")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/05_actual_vs_predicted.png", dpi=150)
plt.close()

# (c) Feature importance (from Random Forest - easiest to explain)
rf = models["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values()
plt.figure(figsize=(7, 4.5))
imp.plot(kind="barh", color="#55A868")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("outputs/06_feature_importance.png", dpi=150)
plt.close()
imp.sort_values(ascending=False).round(3).to_json("outputs/feature_importance.json")
print("\nFeature importance:\n", imp.sort_values(ascending=False).round(3))
print("\nAll outputs saved in outputs/ and models/")
