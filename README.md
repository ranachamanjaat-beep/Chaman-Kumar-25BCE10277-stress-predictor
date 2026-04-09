# 🧠 Personal AI Stress & Productivity Predictor

A Streamlit web application that uses **Linear Regression** (Scikit-Learn) to predict your daily stress score based on four lifestyle habits — sleep, caffeine, work hours, and physical activity. Designed around AI course concepts **CO3** (Normal Distribution) and **CO4** (Supervised Learning, MSE, R²).

---

## 📸 Features at a Glance

| Section | Description |
|---|---|
| **Sidebar Inputs** | Sliders & inputs for daily habit tracking |
| **Dataset Management** | Auto-generates 50-day synthetic seed data via Normal Distribution |
| **AI Model (CO4)** | Linear Regression retrained on every load with MSE & R² metrics |
| **Prediction Engine** | Real-time stress score + feature importance bar chart |
| **Trend Chart** | Stress over time with Training / Validation set visualisation |

---

## 🗂️ Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── stress_data.csv         # Auto-generated on first run (do not delete)
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone or download the project

```bash
git clone git@github.com:ranachamanjaat-beep/Chaman-Kumar-25BCE10277-stress-predictor.git

cd stress-predictor
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.32.0 | Web UI framework |
| `pandas` | ≥ 2.0.0 | Data loading & manipulation |
| `numpy` | ≥ 1.26.0 | Numerical operations & Normal Distribution |
| `scikit-learn` | ≥ 1.4.0 | Linear Regression, StandardScaler, metrics |
| `matplotlib` | ≥ 3.8.0 | Trend chart & feature importance bar chart |

Install all at once:

```bash
pip install -r requirements.txt
```

---

## 🔢 How It Works

### Section 1 — Data Input (Sidebar)

Use the sidebar controls to enter today's habits:

- 😴 **Hours Slept** — slider from 4 to 10 hours
- ☕ **Caffeine Intake** — number input in mg (e.g. espresso ≈ 63 mg, drip coffee ≈ 95 mg)
- 📚 **Study / Work Hours** — slider from 0 to 12 hours
- 🏃 **Physical Activity** — slider from 0 to 120 minutes

Press **💾 Save Today's Entry** to append the day's data and predicted stress score to `stress_data.csv`.

---

### Section 2 — Dataset Management (CO3)

On first launch, if `stress_data.csv` does not exist, the app automatically generates **50 days of synthetic seed data** using a **Normal Distribution** so the model has enough data to train on immediately.

```python
hours_slept       ~ N(μ=7.0, σ=1.2)   clipped to [4, 10]
caffeine_mg       ~ N(μ=180, σ=60)    clipped to [0, 500]
study_work_hours  ~ N(μ=6.0, σ=2.0)   clipped to [0, 12]
physical_activity ~ N(μ=30,  σ=20)    clipped to [0, 120]
```

Stress scores are derived from a weighted formula with added Gaussian noise to simulate real-world variability. Every time you save an entry, the new row is appended to the CSV and the model retrains.

---

### Section 3 — AI Logic (CO4)

The app trains a **Scikit-Learn `LinearRegression`** model on every load:

1. Features are normalised with `StandardScaler`
2. Data is split **80% Training / 20% Validation** (`random_state=42`)
3. The model reports four performance metrics:

| Metric | Set | Interpretation |
|---|---|---|
| **MSE** | Training | Mean Squared Error — lower is better |
| **MSE** | Validation | Measures generalisation |
| **R²** | Training | Fit quality — 1.0 is perfect |
| **R²** | Validation | Generalisation quality |

---

### Section 4 — Prediction Engine

Based on the sidebar inputs, the model predicts a **Stress Score from 1 to 10**:

| Score Range | Status |
|---|---|
| 1.0 – 3.5 | 🟢 LOW — You're doing well! |
| 3.6 – 6.0 | 🟡 MODERATE — Stay mindful. |
| 6.1 – 8.0 | 🟠 HIGH — Take a break. |
| 8.1 – 10.0 | 🔴 CRITICAL — Rest & recover! |

A **Feature Importance bar chart** (horizontal) displays the standardised Linear Regression coefficients for each habit — green bars reduce stress, red bars raise it.

---

## 📊 Visualisations

### Stress Trend Over Time
- Scatter plot of all historical stress scores
- Green region = Training set (80%), orange region = Validation set (20%)
- Dashed line shows 5-day rolling average
- Horizontal marker shows today's predicted stress

### Feature Importance Chart
- Horizontal bar chart of standardised model coefficients
- Values rendered inside each bar for readability
- Positive coefficient (red) → habit increases stress
- Negative coefficient (green) → habit reduces stress

---

## 💾 Data Format

`stress_data.csv` columns:

| Column | Type | Description |
|---|---|---|
| `date` | string | Entry date (YYYY-MM-DD) |
| `hours_slept` | float | Hours of sleep |
| `caffeine_mg` | float | Caffeine in milligrams |
| `study_work_hours` | float | Focused work/study hours |
| `physical_activity_min` | float | Exercise in minutes |
| `stress_score` | float | Target variable (1–10) |

---

## 🧪 Course Concept Mapping

| Concept | Implementation |
|---|---|
| **CO3 — Normal Distribution** | Synthetic seed data generation via `np.random.normal()` |
| **CO4 — Linear Regression** | `sklearn.linear_model.LinearRegression` |
| **CO4 — Feature Scaling** | `sklearn.preprocessing.StandardScaler` |
| **CO4 — Train/Val Split** | `sklearn.model_selection.train_test_split` (80/20) |
| **CO4 — MSE** | `sklearn.metrics.mean_squared_error` |
| **CO4 — R²** | `sklearn.metrics.r2_score` |

---

## 🛠️ Customisation Tips

- **Add more features** — extend `FEATURES` list in `app.py` and update the sidebar inputs and synthetic data generator accordingly.
- **Change the train/val split** — modify `test_size=0.20` in `train_model()`.
- **Swap the model** — replace `LinearRegression()` with any Scikit-Learn regressor (e.g. `Ridge`, `RandomForestRegressor`) — the rest of the pipeline stays the same.
- **Reset data** — delete `stress_data.csv` and relaunch to regenerate fresh synthetic seed data.

---

*Built with Streamlit · Scikit-Learn · Matplotlib · NumPy · Pandas*# chaman-kumar-25BCE10277-stress-predictor
This project is a Streamlit-based AI tool that predicts daily stress levels using inputs like sleep, caffeine intake, work hours, and physical activity. It uses a Linear Regression model trained on synthetic data, providing real-time predictions, performance metrics, and visual insights to help users track and manage stress.
