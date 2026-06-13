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
    sun:     { emoji: '☀️',  message: 'Stay hydrated and avoid too much heat.' },
    rain:    { emoji: '🌧️',  message: 'Carry an umbrella and travel carefully.' },
    fog:     { emoji: '🌫️',  message: 'Drive slowly because visibility may be low.' },
    drizzle: { emoji: '🌦️',  message: 'Carry light rain protection.' },
    snow:    { emoji: '❄️',  message: 'Wear warm clothes and avoid risky travel.' },
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

// ── Audio Engine & Animations ──────────────────────────────
const audioMap = {
    'rain-mode': 'https://actions.google.com/sounds/v1/weather/rain_heavy_loud.ogg',
    'sun-mode': 'https://actions.google.com/sounds/v1/ambiences/birds_in_forest.ogg',
    'snow-mode': 'https://actions.google.com/sounds/v1/weather/winter_wind.ogg',
    'fog-mode': 'https://actions.google.com/sounds/v1/ambiences/creepy_wind.ogg',
    'drizzle-mode': 'https://actions.google.com/sounds/v1/weather/rain_on_roof.ogg'
};

let currentAudio = null;
let isMuted = false;
const muteBtn = document.getElementById('mute-btn');

if (muteBtn) {
    muteBtn.addEventListener('click', () => {
        isMuted = !isMuted;
        muteBtn.textContent = isMuted ? '🔇 Sound Off' : '🔊 Sound On';
        if (currentAudio) currentAudio.muted = isMuted;
    });
}

function playSoundForWeather(weatherClass) {
    if (currentAudio) {
        currentAudio.pause();
        currentAudio.currentTime = 0;
    }
    if (audioMap[weatherClass]) {
        currentAudio = new Audio(audioMap[weatherClass]);
        currentAudio.loop = true;
        currentAudio.muted = isMuted;
        currentAudio.play().catch(e => console.log('Audio autoplay blocked', e));
    }
}

const canvas = document.getElementById('weather-canvas');
const ctx = canvas ? canvas.getContext('2d') : null;
let particles = [];
let animId;

function resize() {
    if (!canvas) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}
window.addEventListener('resize', resize);
resize();

function spawnRain(count, config) {
    particles = [];
    for (let i = 0; i < count; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            len: config.len[0] + Math.random() * config.len[1],
            speed: config.speed[0] + Math.random() * config.speed[1],
            alpha: config.alpha[0] + Math.random() * config.alpha[1],
            width: config.width,
        });
    }
}

function drawRain(color) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.strokeStyle = color;
    particles.forEach(p => {
        ctx.globalAlpha = p.alpha;
        ctx.lineWidth = p.width;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - p.len * 0.15, p.y + p.len);
        ctx.stroke();
        p.y += p.speed;
        if (p.y > canvas.height) { p.y = -p.len; p.x = Math.random() * canvas.width; }
    });
    ctx.restore();
}

function spawnSnow(count) {
    particles = [];
    for (let i = 0; i < count; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            r: 2 + Math.random() * 5,
            speed: 0.4 + Math.random() * 1.2,
            drift: (Math.random() - 0.5) * 0.4,
            alpha: 0.55 + Math.random() * 0.45,
            wobble: Math.random() * Math.PI * 2,
        });
    }
}

function drawSnow() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
        ctx.save();
        ctx.globalAlpha = p.alpha;
        ctx.fillStyle = '#fff';
        ctx.shadowColor = '#c8e6ff';
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        p.wobble += 0.015;
        p.x += p.drift + Math.sin(p.wobble) * 0.3;
        p.y += p.speed;
        if (p.y > canvas.height) { p.y = -10; p.x = Math.random() * canvas.width; }
    });
}

function loop(drawFn, color) {
    drawFn(color);
    animId = requestAnimationFrame(() => loop(drawFn, color));
}

function applyWeatherState(weatherClass) {
    document.body.className = weatherClass;
    document.body.dataset.weather = weatherClass;
    
    if (animId) cancelAnimationFrame(animId);
    if (canvas) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvas.style.display = 'none';
    }

    if (weatherClass === 'rain-mode') {
        canvas.style.display = 'block';
        spawnRain(200, { len:[18,24], speed:[12,10], alpha:[0.25,0.4], width:1.2 });
        loop(drawRain, 'rgba(174,214,241,1)');
    } else if (weatherClass === 'drizzle-mode') {
        canvas.style.display = 'block';
        spawnRain(120, { len:[8,10], speed:[5,4], alpha:[0.18,0.25], width:0.8 });
        loop(drawRain, 'rgba(200,230,255,1)');
    } else if (weatherClass === 'snow-mode') {
        canvas.style.display = 'block';
        spawnSnow(160);
        loop(drawSnow);
    }
    
    playSoundForWeather(weatherClass);
    document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ── Form submit handler ───────────────────────────────────
const loadingMessages = [
    "Analyzing weather data...",
    "Processing machine learning model...",
    "Generating weather prediction...",
    "Preparing visual animation..."
];

document.getElementById('predict-form').addEventListener('submit', async function (e) {
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

    const predictBtn = document.getElementById('predict-btn');
    const loadingContainer = document.getElementById('loading-container');
    const loadingText = document.getElementById('loading-text');
    const resultDiv = document.getElementById('result');

    predictBtn.disabled = true;
    resultDiv.style.display = 'none';
    loadingContainer.style.display = 'block';

    let msgIndex = 0;
    loadingText.textContent = loadingMessages[msgIndex];
    const msgInterval = setInterval(() => {
        msgIndex++;
        if (msgIndex < loadingMessages.length) {
            loadingText.textContent = loadingMessages[msgIndex];
        }
    }, 800);

    // Predict
    const features = computeFeatures(date, precipitation, temp_max, temp_min, wind);
    const { label, confidence, probabilities } = predictWeather(features);

    // Wait until all messages are shown (approx 3.2 seconds total)
    const minWait = loadingMessages.length * 800;
    await new Promise(resolve => setTimeout(resolve, minWait));

    clearInterval(msgInterval);
    loadingContainer.style.display = 'none';
    predictBtn.disabled = false;

    showResult(label, confidence, probabilities);
    applyWeatherState(label + "-mode");
});
