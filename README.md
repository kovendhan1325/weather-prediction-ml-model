# 🌤 Weather Prediction ML Model

[![Live Demo](https://img.shields.io/badge/Live%20Demo-GitHub%20Pages-blue?style=for-the-badge&logo=github)](https://Kovendhan1325.github.io/weather-prediction-ml-model/)

A machine learning weather prediction web application trained on the **Seattle Weather Dataset** (1461 daily records, 2012–2015). Predicts weather as **rain, sun, fog, drizzle, or snow** from temperature, precipitation, and wind inputs.

---

## 🚀 Live Demo

👉 **[https://Kovendhan1325.github.io/weather-prediction-ml-model/](https://Kovendhan1325.github.io/weather-prediction-ml-model/)**

The GitHub Pages demo runs entirely in the browser — no server required.

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask |
| ML | Scikit-learn, Pandas, NumPy |
| Visualisation | Matplotlib |
| Frontend | HTML5, CSS3 (Glassmorphism), Vanilla JS |
| Deployment | GitHub Pages (static) / Flask (local) |

---

## 📊 Model Performance

| Model | Test Accuracy |
|---|---|
| **Random Forest** ✅ | **84.64%** |
| Extra Trees | 84.64% |
| Gradient Boosting | 84.30% |
| SVM | 71.67% |
| Logistic Regression | 62.80% |

---

## 📁 Dataset Columns

| Column | Description |
|---|---|
| `date` | Date of observation |
| `precipitation` | Precipitation in mm |
| `temp_max` | Maximum temperature °C |
| `temp_min` | Minimum temperature °C |
| `wind` | Wind speed km/h |
| `weather` | Target label (rain/sun/fog/drizzle/snow) |

---

## ⚙️ Run Locally (Flask)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train_model.py

# 3. Start the server
python app.py
```

Then open **http://127.0.0.1:5000/**

---

## 🔧 Features Engineered

Beyond the raw inputs, the model uses:
- Date decomposition: `year`, `month`, `day`, `dayofyear`, `weekofyear`
- Temperature features: `temp_range`, `temp_avg`, `freezing_or_below`
- Precipitation flags: `has_precipitation`, `precip_intensity` (binned)
- Wind category: `wind_category` (binned into 4 levels)
- Season indicator

---

## 📂 Project Structure

```
weather_prediction_project/
├── app.py                  # Flask web server
├── train_model.py          # Model training script
├── seattle-weather.csv     # Dataset
├── requirements.txt        # Python dependencies
├── static/
│   ├── style.css           # Flask app styles
│   └── weather_bg.png      # Background image
├── templates/
│   └── index.html          # Flask Jinja2 template
└── docs/                   # GitHub Pages static site
    ├── index.html
    ├── style.css
    ├── predict.js           # Client-side prediction engine
    └── weather_bg.png
```

---

## 📄 License

MIT — free to use, modify, and distribute.