"""
export_model_js.py
Trains a compact GradientBoosting model and exports it to JavaScript
using m2cgen so the prediction runs entirely in the browser (GitHub Pages).
"""

import os
import json
import pandas as pd
import m2cgen as m2c
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Load & engineer features (mirrors train_model.py) ────────
df = pd.read_csv("seattle-weather.csv")
df = df.dropna()
df["date"] = pd.to_datetime(df["date"])

df["year"]       = df["date"].dt.year
df["month"]      = df["date"].dt.month
df["day"]        = df["date"].dt.day
df["dayofyear"]  = df["date"].dt.dayofyear
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
df["temp_range"]        = df["temp_max"] - df["temp_min"]
df["temp_avg"]          = (df["temp_max"] + df["temp_min"]) / 2
df["has_precipitation"] = (df["precipitation"] > 0).astype(int)
df["freezing_or_below"] = (df["temp_min"] <= 0).astype(int)
df["season"]            = df["month"] % 12 // 3 + 1

df["wind_category"] = pd.cut(
    df["wind"],
    bins=[0, 2, 4, 6, float("inf")],
    labels=[0, 1, 2, 3],
    include_lowest=True
).astype(int)

df["precip_intensity"] = pd.cut(
    df["precipitation"],
    bins=[-0.001, 0, 5, 20, float("inf")],
    labels=[0, 1, 2, 3]
).astype(int)

features = [
    "precipitation", "temp_max", "temp_min", "wind",
    "year", "month", "day", "dayofyear", "weekofyear",
    "temp_range", "temp_avg", "has_precipitation",
    "freezing_or_below", "season", "wind_category", "precip_intensity",
]

X = df[features]
y = df["weather"]
class_names = sorted(y.unique().tolist())

# ── Train a compact model specifically for JS export ─────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

export_model = GradientBoostingClassifier(
    n_estimators=20,
    max_depth=3,
    learning_rate=0.15,
    random_state=42,
)
export_model.fit(X_train, y_train)
acc = accuracy_score(y_test, export_model.predict(X_test))
print(f"Export model test accuracy: {acc * 100:.2f}%")

# Refit on full dataset for best generalisation
export_model.fit(X, y)

# ── Export to JavaScript ──────────────────────────────────────
os.makedirs("docs", exist_ok=True)
raw_js = m2c.export_to_javascript(export_model)

js_output = f"""// AUTO-GENERATED — do not edit manually
// Compact GradientBoosting model exported via m2cgen
// Test accuracy: {acc * 100:.2f}%
// Classes (index order): {json.dumps(class_names)}

const MODEL_CLASSES = {json.dumps(class_names)};

{raw_js}
"""

with open("docs/model.js", "w", encoding="utf-8") as f:
    f.write(js_output)

size_kb = os.path.getsize("docs/model.js") / 1024
print(f"docs/model.js written ({size_kb:.0f} KB)")
print("Classes:", class_names)
