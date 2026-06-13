// predict.js — client-side weather prediction for GitHub Pages
// Rule-based predictor calibrated against the Seattle weather dataset
// Mirrors the feature engineering from train_model.py

// ── ISO week-of-year helper ────────────────────────────────
function getISOWeek(date) {
    const d = new Date(Date.UTC(date.getFullYear(), date.getMonth(), date.getDate()));
    d.setUTCDate(d.getUTCDate() + 4 - (d.getUTCDay() || 7));
    const yearStart = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    return Math.ceil((((d - yearStart) / 86400000) + 1) / 7);
}

// ── Feature engineering (mirrors train_model.py) ───────────
function computeFeatures(dateStr, precipitation, temp_max, temp_min, wind) {
    const d     = new Date(dateStr);
    const year  = d.getFullYear();
    const month = d.getMonth() + 1; // 1-12
    const day   = d.getDate();

    const start      = new Date(year, 0, 0);
    const dayofyear  = Math.round((d - start) / 86400000);
    const weekofyear = getISOWeek(d);

    const temp_range        = temp_max - temp_min;
    const temp_avg          = (temp_max + temp_min) / 2;
    const has_precipitation = precipitation > 0 ? 1 : 0;
    const freezing_or_below = temp_min <= 0 ? 1 : 0;
    const season            = Math.floor(month % 12 / 3) + 1; // 1=winter 2=spring 3=summer 4=fall

    // Wind category bins: [0,2] [2,4] [4,6] [6+]
    const wind_category =
        wind <= 2 ? 0 : wind <= 4 ? 1 : wind <= 6 ? 2 : 3;

    // Precip intensity bins: [0,0] [0,5] [5,20] [20+]
    const precip_intensity =
        precipitation <= 0 ? 0 : precipitation <= 5 ? 1 : precipitation <= 20 ? 2 : 3;

    return {
        precipitation, temp_max, temp_min, wind,
        year, month, day, dayofyear, weekofyear,
        temp_range, temp_avg,
        has_precipitation, freezing_or_below, season,
        wind_category, precip_intensity
    };
}

// ── Weather prediction (decision-rule model) ──────────────
// Rules trained on the patterns identified by the Random Forest / GB models.
// Returns { label, probabilities } where probabilities is an object keyed by class.
function predictWeather(f) {
    // Raw score accumulators per class
    const score = { drizzle: 0, fog: 0, rain: 0, snow: 0, sun: 0 };

    // ── SNOW ─────────────────────────────────────────────────
    if (f.temp_min <= 0) {
        score.snow     += 4.0;
        score.rain     -= 1.0;
        score.sun      -= 2.0;
    }
    if (f.temp_min <= -1 && f.precipitation > 1) score.snow += 3.0;
    if (f.freezing_or_below && f.precip_intensity >= 1) score.snow += 2.0;

    // ── RAIN ─────────────────────────────────────────────────
    if (f.precipitation > 10) score.rain += 5.0;
    if (f.precipitation > 5)  score.rain += 3.0;
    if (f.precipitation > 2 && f.temp_avg < 16) score.rain += 2.5;
    if (f.precip_intensity === 2 || f.precip_intensity === 3) score.rain += 2.0;
    if (f.season !== 3 && f.has_precipitation) score.rain += 1.0; // not summer
    if (f.month >= 10 || f.month <= 3) score.rain += 1.5;        // Oct–Mar rainy season

    // ── DRIZZLE ──────────────────────────────────────────────
    if (f.precipitation > 0 && f.precipitation <= 5 && f.temp_avg > 5) {
        score.drizzle += 2.5;
    }
    if (f.precip_intensity === 1 && f.temp_avg < 14) score.drizzle += 2.0;
    if (f.wind_category <= 1 && f.precipitation > 0 && f.precipitation < 4) score.drizzle += 1.5;

    // ── FOG ──────────────────────────────────────────────────
    if (f.precipitation < 1 && f.temp_max < 13 && f.wind_category <= 1) score.fog += 3.5;
    if (f.temp_avg < 8  && f.wind_category === 0 && !f.has_precipitation)  score.fog += 2.5;
    if (f.temp_range < 5 && f.temp_avg < 12 && f.wind_category === 0)      score.fog += 2.0;
    if (f.month >= 11 || f.month <= 2) score.fog += 1.0; // fog more common in winter

    // ── SUN ──────────────────────────────────────────────────
    if (!f.has_precipitation && f.temp_avg > 12) score.sun += 4.0;
    if (!f.has_precipitation && f.temp_max > 18) score.sun += 3.0;
    if (f.season === 3)                          score.sun += 3.5; // summer
    if (f.month >= 6 && f.month <= 9)            score.sun += 2.0;
    if (f.precipitation === 0 && f.wind_category >= 1) score.sun += 1.5;
    if (!f.has_precipitation && f.temp_range > 8)      score.sun += 1.5; // high diurnal range

    // ── Softmax to get probabilities ─────────────────────────
    const classes = ['drizzle', 'fog', 'rain', 'snow', 'sun'];
    const vals    = classes.map(c => score[c]);
    const maxVal  = Math.max(...vals);
    const expVals = vals.map(v => Math.exp(v - maxVal));
    const sumExp  = expVals.reduce((a, b) => a + b, 0);
    const proba   = {};
    classes.forEach((c, i) => { proba[c] = expVals[i] / sumExp; });

    // Sort probabilities descending
    const sorted = Object.entries(proba).sort((a, b) => b[1] - a[1]);
    const label  = sorted[0][0];
    const sortedProba = Object.fromEntries(sorted);

    return { label, confidence: sorted[0][1], probabilities: sortedProba };
}

// ── Weather metadata ──────────────────────────────────────
const WEATHER_META = {
    sun:     { emoji: '☀️',  message: 'Sunny weather is expected. Great day for outdoor activities!' },
    rain:    { emoji: '🌧️',  message: 'Rainy weather expected. Carry an umbrella and travel carefully.' },
    fog:     { emoji: '🌫️',  message: 'Foggy conditions expected. Visibility may be low — drive carefully.' },
    drizzle: { emoji: '🌦️',  message: 'Light drizzle expected. A light jacket or umbrella may help.' },
    snow:    { emoji: '❄️',  message: 'Snowy weather expected. Stay warm and avoid risky travel.' },
};

// ── Input validation ──────────────────────────────────────
function validateInputs(precipitation, temp_max, temp_min, wind) {
    const errors = [];
    if (precipitation < 0 || precipitation > 200)
        errors.push('Precipitation must be between 0 – 200 mm.');
    if (temp_max < -20 || temp_max > 50)
        errors.push('Max temperature must be between -20°C and 50°C.');
    if (temp_min < -20 || temp_min > 50)
        errors.push('Min temperature must be between -20°C and 50°C.');
    if (temp_min > temp_max)
        errors.push('Min temperature cannot exceed max temperature.');
    if (wind < 0 || wind > 150)
        errors.push('Wind speed must be between 0 – 150 km/h.');
    return errors;
}

// ── DOM helpers ───────────────────────────────────────────
function showResult(label, confidence, probabilities, isError = false) {
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'block';

    if (isError) {
        resultDiv.className = 'result result-error';
        document.getElementById('result-title').textContent = label;
        document.getElementById('result-msg').textContent   = confidence; // reused as msg
        document.getElementById('confidence-wrap').style.display  = 'none';
        document.getElementById('breakdown-wrap').style.display   = 'none';
        return;
    }

    const meta = WEATHER_META[label] || { emoji: '🌍', message: 'Weather predicted.' };
    resultDiv.className = 'result';

    document.getElementById('result-title').textContent =
        `${meta.emoji} Predicted Weather: ${label.toUpperCase()}`;
    document.getElementById('result-msg').textContent   = meta.message;

    // Confidence badge
    const pct = (confidence * 100).toFixed(1);
    document.getElementById('confidence-text').textContent = `Model confidence: ${pct}%`;
    document.getElementById('confidence-wrap').style.display = 'flex';

    // Probability bars
    const breakdown = document.getElementById('breakdown-bars');
    breakdown.innerHTML = '';
    Object.entries(probabilities).forEach(([cls, prob], idx) => {
        const p = (prob * 100).toFixed(1);
        const row = document.createElement('div');
        row.className = 'conf-row';
        row.innerHTML = `
            <span class="conf-label">${cls}</span>
            <div class="conf-bar-wrap">
                <div class="conf-bar-fill ${idx === 0 ? 'conf-bar-top' : ''}"
                     style="--pct:${p}%"></div>
            </div>
            <span class="conf-pct">${p}%</span>`;
        breakdown.appendChild(row);
    });
    document.getElementById('breakdown-wrap').style.display = 'block';

    // Trigger bar animation
    requestAnimationFrame(() => {
        document.querySelectorAll('.conf-bar-fill').forEach(el => {
            el.style.animation = 'none';
            void el.offsetWidth; // reflow
            el.style.animation = '';
        });
    });
}

// ── Form submit handler ───────────────────────────────────
document.getElementById('predict-form').addEventListener('submit', function (e) {
    e.preventDefault();

    const date          = document.getElementById('date').value;
    const precipitation = parseFloat(document.getElementById('precipitation').value);
    const temp_max      = parseFloat(document.getElementById('temp_max').value);
    const temp_min      = parseFloat(document.getElementById('temp_min').value);
    const wind          = parseFloat(document.getElementById('wind').value);

    const errors = validateInputs(precipitation, temp_max, temp_min, wind);
    if (errors.length > 0) {
        showResult('⚠️ Invalid Input', errors.join(' '), {}, true);
        return;
    }

    const features = computeFeatures(date, precipitation, temp_max, temp_min, wind);
    const { label, confidence, probabilities } = predictWeather(features);
    showResult(label, confidence, probabilities);
});
