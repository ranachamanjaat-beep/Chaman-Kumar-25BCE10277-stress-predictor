import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI Stress & Productivity Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');
  html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
  h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; letter-spacing: -0.5px; }
  .stApp { background: #0d0f14; color: #e8e8e8; }
  .block-container { max-width: 100% !important; padding-left: clamp(1rem, 3vw, 3rem) !important; padding-right: clamp(1rem, 3vw, 3rem) !important; padding-top: 3rem !important; }
  [data-testid="metric-container"] { background: #12151d; border: 1px solid #1e2235; border-radius: 12px; padding: 1rem 1.2rem; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #c8f0a0 !important; font-family: 'Syne', sans-serif !important; font-size: 2rem !important; font-weight: 800; }
  .section-header { font-family: 'Syne', sans-serif; font-size: 0.65rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: #4a6fa5; margin-bottom: 0.3rem; padding-bottom: 0.5rem; border-bottom: 1px solid #1a1f2e; }
  .stress-badge { background: linear-gradient(135deg, #1a2540 0%, #0e1825 100%); border: 1px solid #2a3a5e; border-radius: 16px; padding: 1.5rem 2rem; text-align: center; }
  .stress-score-number { font-family: 'Syne', sans-serif; font-size: clamp(2.5rem, 8vw, 4.5rem); font-weight: 800; }
  .pill { display: inline-block; background: #151a28; border: 1px solid #232a40; border-radius: 20px; padding: 0.25rem 0.75rem; font-size: 0.72rem; color: #7a8099; margin: 0.2rem; }
  .fancy-divider { height: 1px; background: linear-gradient(90deg, transparent, #2a3a5e, transparent); margin: 1.5rem 0; }
  .tag-train { background: #1a2f1a; color: #7acc7a; border: 1px solid #2a4a2a; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.68rem; text-transform: uppercase; }
  .tag-val { background: #2a1f10; color: #d4944a; border: 1px solid #4a3018; border-radius: 6px; padding: 0.2rem 0.6rem; font-size: 0.68rem; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

DATA_FILE_PATH = "stress_data.csv"
FEATURE_COLUMNS = ["hours_slept", "caffeine_mg", "hours_study", "activity"]

def create_synthetic_history(days_to_generate: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    sleep_patterns = np.random.normal(loc=7.0, scale=1.2, size=days_to_generate).clip(4, 10)
    caffeine_usage = np.random.normal(loc=180, scale=60, size=days_to_generate).clip(0, 500)
    study_habits = np.random.normal(loc=6.0, scale=2.0, size=days_to_generate).clip(0, 12)
    exercise_minutes = np.random.normal(loc=30, scale=20, size=days_to_generate).clip(0, 120)

    calculated_stress_levels = (
        -0.7 * sleep_patterns
        + 0.004 * caffeine_usage
        + 0.35 * study_habits
        - 0.025 * exercise_minutes
        + np.random.normal(0, 0.5, days_to_generate)
        + 5.5
    ).clip(1, 10)

    timeline = [datetime.today() - timedelta(days=days_to_generate - i) for i in range(days_to_generate)]

    historical_records = pd.DataFrame({
        "date": [day.strftime("%Y-%m-%d") for day in timeline],
        "hours_slept": np.round(sleep_patterns, 2),
        "caffeine_mg": np.round(caffeine_usage, 1),
        "hours_study": np.round(study_habits, 2),
        "activity": np.round(exercise_minutes, 1),
        "stress_score": np.round(calculated_stress_levels, 2),
    })
    return historical_records

def load_user_data():
    required_columns = FEATURE_COLUMNS + ["stress_score"]

    if not os.path.exists(DATA_FILE_PATH):
        initial_data = create_synthetic_history(50)
        initial_data.to_csv(DATA_FILE_PATH, index=False)
        return initial_data, True

    user_df = pd.read_csv(DATA_FILE_PATH)

    standard_column_names = {
        "physical_activity_min": "activity",
        "physical_activity": "activity",
        "study_hours": "hours_study",
        "sleep": "hours_slept",
        "caffeine": "caffeine_mg",
        "stress": "stress_score",
    }
    user_df.rename(columns={old: new for old, new in standard_column_names.items() if old in user_df.columns}, inplace=True)

    missing_fields = [col for col in required_columns if col not in user_df.columns]
    if missing_fields:
        rebuilt_data = create_synthetic_history(50)
        rebuilt_data.to_csv(DATA_FILE_PATH, index=False)
        return rebuilt_data, True

    return user_df, False

def save_daily_snapshot(sleep_amount, caffeine_amount, study_time, physical_activity, predicted_stress):
    current_dataset, _ = load_user_data()
    new_entry = pd.DataFrame([{
        "date": datetime.today().strftime("%Y-%m-%d"),
        "hours_slept": sleep_amount,
        "caffeine_mg": caffeine_amount,
        "hours_study": study_time,
        "activity": physical_activity,
        "stress_score": round(predicted_stress, 2),
    }])
    updated_dataset = pd.concat([current_dataset, new_entry], ignore_index=True)
    updated_dataset.to_csv(DATA_FILE_PATH, index=False)
    return updated_dataset

@st.cache_data(ttl=30)
def train_predictive_engine(csv_path: str):
    raw_data = pd.read_csv(csv_path)
    
    input_features = raw_data[FEATURE_COLUMNS].values
    target_output = raw_data["stress_score"].values

    x_train, x_test, y_train, y_test = train_test_split(
        input_features, target_output, test_size=0.20, random_state=42
    )

    feature_scaler = StandardScaler()
    x_train_scaled = feature_scaler.fit_transform(x_train)
    x_test_scaled = feature_scaler.transform(x_test)

    regression_model = LinearRegression()
    regression_model.fit(x_train_scaled, y_train)

    training_predictions = regression_model.predict(x_train_scaled)
    testing_predictions = regression_model.predict(x_test_scaled)

    model_metrics = {
        "mse_train": mean_squared_error(y_train, training_predictions),
        "mse_val": mean_squared_error(y_test, testing_predictions),
        "r2_train": r2_score(y_train, training_predictions),
        "r2_val": r2_score(y_test, testing_predictions),
        "n_train": len(x_train),
        "n_val": len(x_test),
    }
    return regression_model, feature_scaler, model_metrics, raw_data

with st.sidebar:
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.25rem;font-weight:800;color:#c8f0a0;">🧠 Daily Inputs</div>', unsafe_allow_html=True)

    user_sleep_input = st.slider("😴 Hours Slept", 4.0, 10.0, 7.0, 0.5)
    user_caffeine_input = st.number_input("☕ Caffeine Intake (mg)", 0, 600, 180, 10)
    user_study_input = st.slider("📚 Study / Work Hours", 0.0, 12.0, 6.0, 0.5)
    user_activity_input = st.slider("🏃 Physical Activity (min)", 0, 120, 30, 5)

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    save_trigger = st.button("💾 Save Today's Entry", use_container_width=True)

current_history_df, has_reset = load_user_data()
if has_reset:
    st.toast("✨ 50 synthetic seed days generated", icon="🌱")

model_engine, data_scaler, evaluation_metrics, full_history = train_predictive_engine(DATA_FILE_PATH)

today_stats = np.array([[user_sleep_input, user_caffeine_input, user_study_input, user_activity_input]])
today_stats_scaled = data_scaler.transform(today_stats)
predicted_stress_level = float(model_engine.predict(today_stats_scaled)[0])
predicted_stress_level = round(np.clip(predicted_stress_level, 1.0, 10.0), 2)

def determine_ui_theme(stress_val):
    if stress_val <= 3.5: return "#7acc7a", "LOW — You're doing well!"
    if stress_val <= 6.0: return "#f0c060", "MODERATE — Stay mindful."
    if stress_val <= 8.0: return "#e0884a", "HIGH — Take a break."
    return "#e05555", "CRITICAL — Rest & recover!"

theme_color, feedback_message = determine_ui_theme(predicted_stress_level)

if save_trigger:
    save_daily_snapshot(user_sleep_input, user_caffeine_input, user_study_input, user_activity_input, predicted_stress_level)
    st.cache_data.clear()
    st.rerun()

st.markdown("""
<div style="margin-bottom:1.5rem">
  <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#e8eaf0;line-height:1.1;">Personal AI Stress Predictor</div>
  <div style="font-size:0.72rem;color:#3a4565;letter-spacing:2.5px;text-transform:uppercase;">Linear Regression · Scikit-Learn</div>
</div>
""", unsafe_allow_html=True)

prediction_col, analysis_col = st.columns([1, 2], gap="large")

with prediction_col:
    st.markdown('<div class="section-header">Today\'s Prediction</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stress-badge">
      <div class="stress-label">Stress Score</div>
      <div class="stress-score-number" style="color:{theme_color};">{predicted_stress_level}</div>
      <div style="font-size:0.68rem;letter-spacing:2px;color:{theme_color};">{feedback_message}</div>
      <div style="margin-top:1rem;display:grid;grid-template-columns:1fr 1fr;gap:0.35rem;">
        <span class="pill">😴 {user_sleep_input}h</span>
        <span class="pill">☕ {user_caffeine_input}mg</span>
        <span class="pill">📚 {user_study_input}h</span>
        <span class="pill">🏃 {user_activity_input}m</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with analysis_col:
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    metric_l, metric_r = st.columns(2)
    metric_l.metric("MSE Training", f"{evaluation_metrics['mse_train']:.4f}")
    metric_l.metric("R² Training", f"{evaluation_metrics['r2_train']:.4f}")
    metric_r.metric("MSE Validation", f"{evaluation_metrics['mse_val']:.4f}")
    metric_r.metric("R² Validation", f"{evaluation_metrics['r2_val']:.4f}")

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
trend_col, importance_col = st.columns([3, 2], gap="large")

with trend_col:
    st.markdown('<div class="section-header">Stress Trend Over Time</div>', unsafe_allow_html=True)
    trend_df = full_history.copy()
    trend_df["date"] = pd.to_datetime(trend_df["date"])
    trend_df = trend_df.sort_values("date").reset_index(drop=True)
    
    line_fig, line_ax = plt.subplots(figsize=(9, 4), facecolor="#0e1118")
    line_ax.set_facecolor("#0e1118")
    line_ax.plot(trend_df.index, trend_df["stress_score"], color="#7acc7a", lw=1.5, alpha=0.6)
    line_ax.scatter(trend_df.index, trend_df["stress_score"], color="#7acc7a", s=15)
    line_ax.set_ylim(0, 11)
    line_ax.tick_params(colors="#8090b0", labelsize=7)
    for spine in line_ax.spines.values(): spine.set_edgecolor("#1a1f2e")
    st.pyplot(line_fig, use_container_width=True)

with importance_col:
    st.markdown('<div class="section-header">Feature Impact</div>', unsafe_allow_html=True)
    impact_weights = model_engine.coef_
    bar_fig, bar_ax = plt.subplots(figsize=(7, 5), facecolor="#0e1118")
    bar_ax.set_facecolor("#0e1118")
    bar_ax.barh(FEATURE_COLUMNS, impact_weights, color=["#7acc7a" if weight < 0 else "#e05555" for weight in impact_weights])
    bar_ax.tick_params(colors="#8090b0", labelsize=8)
    for spine in bar_ax.spines.values(): spine.set_edgecolor("#1a1f2e")
    st.pyplot(bar_fig, use_container_width=True)

with st.expander("📋 Data Log", expanded=False):
    st.dataframe(full_history.tail(15), use_container_width=True)
