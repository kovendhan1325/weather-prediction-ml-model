import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("seattle-weather.csv")

print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nWeather Class Count:")
print(df["weather"].value_counts())

df = df.dropna()
df["date"] = pd.to_datetime(df["date"])

# ── Date features ────────────────────────────────────────────
df["year"]       = df["date"].dt.year
df["month"]      = df["date"].dt.month
df["day"]        = df["date"].dt.day
df["dayofyear"]  = df["date"].dt.dayofyear
df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)

# ── Temperature features ─────────────────────────────────────
df["temp_range"]      = df["temp_max"] - df["temp_min"]
df["temp_avg"]        = (df["temp_max"] + df["temp_min"]) / 2
df["has_precipitation"]  = (df["precipitation"] > 0).astype(int)
df["freezing_or_below"]  = (df["temp_min"] <= 0).astype(int)
df["season"]          = df["month"] % 12 // 3 + 1

# ── Suggestion 5: Additional engineered features ─────────────
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
    "precipitation",
    "temp_max",
    "temp_min",
    "wind",
    "year",
    "month",
    "day",
    "dayofyear",
    "weekofyear",
    "temp_range",
    "temp_avg",
    "has_precipitation",
    "freezing_or_below",
    "season",
    "wind_category",
    "precip_intensity",
]

X = df[features]
y = df["weather"]
class_names = sorted(y.unique().tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=14,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=8,
            gamma="scale",
            class_weight="balanced",
            probability=True,          # needed for predict_proba
        )),
    ]),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ]),
}

best_model      = None
best_accuracy   = 0
best_model_name = ""
best_prediction = None

print("\nModel Accuracy Comparison:")
print("-" * 45)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: {accuracy * 100:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy   = accuracy
        best_model      = model
        best_model_name = name
        best_prediction = y_pred

print("\nBest Model:", best_model_name)
print("Best Accuracy:", round(best_accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, best_prediction))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, best_prediction))

# ── Suggestion 6: 5-Fold Cross-Validation ───────────────────
print("\n5-Fold Cross-Validation on Best Model:")
cv_model  = clone(best_model)
cv_scores = cross_val_score(cv_model, X, y, cv=5, scoring="accuracy")
print(f"CV Accuracy: {cv_scores.mean() * 100:.2f}% ± {cv_scores.std() * 100:.2f}%")

# Refit best model on the full dataset
best_model.fit(X, y)

model_data = {
    "model":       best_model,
    "features":    features,
    "model_name":  best_model_name,
    "accuracy":    best_accuracy,
    "cv_accuracy": float(cv_scores.mean()),
    "cv_std":      float(cv_scores.std()),
    "class_names": class_names,
}

with open("weather_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\nBest model saved successfully as weather_model.pkl")

# ── Suggestion 7: Save chart to static/ so Flask can serve it
weather_counts = df["weather"].value_counts()

palette = ["#3b82f6", "#06b6d4", "#8b5cf6", "#f59e0b", "#10b981"]
fig, ax = plt.subplots(figsize=(9, 5))
weather_counts.plot(
    kind="bar",
    ax=ax,
    color=palette[:len(weather_counts)],
    edgecolor="white",
    width=0.6,
)
ax.set_title("Weather Class Distribution — Seattle Dataset",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xlabel("Weather Type", fontsize=12)
ax.set_ylabel("Count", fontsize=12)
ax.tick_params(axis="x", rotation=0, labelsize=11)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("static/weather_distribution.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print("Weather distribution chart saved as static/weather_distribution.png")