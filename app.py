from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open("weather_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model       = model_data["model"]
features    = model_data["features"]
model_name  = model_data["model_name"]
accuracy    = model_data["accuracy"]
cv_accuracy = model_data.get("cv_accuracy")
cv_std      = model_data.get("cv_std")
class_names = model_data.get("class_names", ["drizzle", "fog", "rain", "snow", "sun"])


# ── Suggestion 9: Input validation ──────────────────────────
def validate_inputs(precipitation, temp_max, temp_min, wind):
    errors = []
    if not (0 <= precipitation <= 200):
        errors.append("Precipitation must be between 0 – 200 mm.")
    if not (-20 <= temp_max <= 50):
        errors.append("Max temperature must be between -20°C and 50°C.")
    if not (-20 <= temp_min <= 50):
        errors.append("Min temperature must be between -20°C and 50°C.")
    if temp_min > temp_max:
        errors.append("Min temperature cannot exceed max temperature.")
    if not (0 <= wind <= 150):
        errors.append("Wind speed must be between 0 – 150 km/h.")
    return errors


def prepare_input(date, precipitation, temp_max, temp_min, wind):
    date = pd.to_datetime(date)

    year       = date.year
    month      = date.month
    day        = date.day
    dayofyear  = date.dayofyear
    weekofyear = int(date.isocalendar().week)

    temp_range        = temp_max - temp_min
    temp_avg          = (temp_max + temp_min) / 2
    has_precipitation = 1 if precipitation > 0 else 0
    freezing_or_below = 1 if temp_min <= 0 else 0
    season            = month % 12 // 3 + 1

    # Suggestion 5: match new engineered features from train_model.py
    wind_category = int(
        pd.cut([wind],
               bins=[0, 2, 4, 6, float("inf")],
               labels=[0, 1, 2, 3],
               include_lowest=True)[0]
    )
    precip_intensity = int(
        pd.cut([precipitation],
               bins=[-0.001, 0, 5, 20, float("inf")],
               labels=[0, 1, 2, 3])[0]
    )

    row = {
        "precipitation":    precipitation,
        "temp_max":         temp_max,
        "temp_min":         temp_min,
        "wind":             wind,
        "year":             year,
        "month":            month,
        "day":              day,
        "dayofyear":        dayofyear,
        "weekofyear":       weekofyear,
        "temp_range":       temp_range,
        "temp_avg":         temp_avg,
        "has_precipitation": has_precipitation,
        "freezing_or_below": freezing_or_below,
        "season":           season,
        "wind_category":    wind_category,
        "precip_intensity": precip_intensity,
    }
    return pd.DataFrame([row])[features]


def common_ctx():
    """Shared template context variables."""
    return dict(
        model_name=model_name,
        accuracy=round(accuracy * 100, 2),
        cv_accuracy=round(cv_accuracy * 100, 2) if cv_accuracy else None,
        cv_std=round(cv_std * 100, 2) if cv_std else None,
    )


@app.route("/")
def home():
    return render_template("index.html", prediction_class="", **common_ctx())


@app.route("/predict", methods=["POST"])
def predict():
    try:
        date          = request.form["date"]
        precipitation = float(request.form["precipitation"])
        temp_max      = float(request.form["temp_max"])
        temp_min      = float(request.form["temp_min"])
        wind          = float(request.form["wind"])

        # Suggestion 9: validate before predicting
        errors = validate_inputs(precipitation, temp_max, temp_min, wind)
        if errors:
            return render_template(
                "index.html",
                prediction_text="⚠️ Invalid Input",
                message=" ".join(errors),
                is_error=True,
                **common_ctx(),
            )

        input_data = prepare_input(date, precipitation, temp_max, temp_min, wind)
        prediction = model.predict(input_data)[0]

        # Suggestion 8: confidence via predict_proba
        confidence  = None
        proba_dict  = None
        try:
            proba      = model.predict_proba(input_data)[0]
            confidence = round(float(max(proba)) * 100, 1)
            # Sort by probability descending for the UI bars
            proba_dict = dict(
                sorted(
                    zip(class_names, [round(float(p) * 100, 1) for p in proba]),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
        except Exception:
            pass

        weather_messages = {
            "sun":     {"emoji": "☀️",  "message": "Stay hydrated and avoid too much heat."},
            "rain":    {"emoji": "🌧️",  "message": "Carry an umbrella and travel carefully."},
            "fog":     {"emoji": "🌫️",  "message": "Drive slowly because visibility may be low."},
            "drizzle": {"emoji": "🌦️",  "message": "Carry light rain protection."},
            "snow":    {"emoji": "❄️",  "message": "Wear warm clothes and avoid risky travel."},
        }

        result          = weather_messages.get(prediction, {"emoji": "🌍", "message": "Weather predicted successfully."})
        prediction_text = f"{result['emoji']} Predicted Weather: {prediction.upper()}"

        # ── Dynamic animation class based on prediction ──────────
        weather_class_map = {
            "rain":    "rain-mode",
            "sun":     "sun-mode",
            "snow":    "snow-mode",
            "drizzle": "drizzle-mode",
            "fog":     "fog-mode",
        }
        prediction_class = weather_class_map.get(prediction, "default-mode")

        return render_template(
            "index.html",
            prediction_text=prediction_text,
            message=result["message"],
            confidence=confidence,
            proba_dict=proba_dict,
            prediction_class=prediction_class,
            **common_ctx(),
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text="❌ Error occurred. Please enter valid input values.",
            message=str(e),
            is_error=True,
            **common_ctx(),
        )


if __name__ == "__main__":
    app.run(debug=True)