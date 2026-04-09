# Project: AI Stress & Productivity Predictor
# Course: Data Science 101 / Machine Learning Basics
# Notes: Using Streamlit for the UI and Scikit-learn for the regression model.
# Goal: Predict stress levels based on daily habits.

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
    initial_sidebar_state="expanded",   # was: ini_sidebar="expanded"
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
  }
  h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.5px;
  }

  .stApp {
    background: #0d0f14;
    color: #e8e8e8;
  }

  .block-container {
    max-width: 100% !important;
    padding-left: clamp(1rem, 3vw, 3rem) !important;
    padding-right: clamp(1rem, 3vw, 3rem) !important;
    padding-top: 3rem !important;
  }
  [data-testid="column"] {
    min-width: 0;
  }

  [data-testid="stSidebar"] {
    background: #111318 !important;
    border-right: 1px solid #1f2230;
  }
  [data-testid="stSidebar"] .stSlider > div > div > div {
    background: #232635;
  }

  [data-testid="metric-container"] {
    background: #12151d;
    border: 1px solid #1e2235;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }
  [data-testid="metric-container"] label {
    color: #7a7f99 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #c8f0a0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem !important;
    font-weight: 800;
  }

  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #4a6fa5;
    margin-bottom: 0.3rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1a1f2e;
  }

  .stress-badge {
    background: linear-gradient(135deg, #1a2540 0%, #0e1825 100%);
    border: 1px solid #2a3a5e;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04);
  }
  .stress-score-number {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.5rem, 8vw, 4.5rem);
    font-weight: 800;
    line-height: 1;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: clip;
  }
  .stress-label {
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #5a6080;
    margin-top: 0.3rem;
  }

  .pill {
    display: inline-block;
    background: #151a28;
    border: 1px solid #232a40;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.72rem;
    color: #7a8099;
    margin: 0.2rem;
  }

  .fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a3a5e, transparent);
    margin: 1.5rem 0;
  }

  .info-row {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid #12151e;
    font-size: 0.78rem;
  }
  .info-key { color: #5a6080; }
  .info-val { color: #c0c8e0; font-weight: 500; }

  .chart-box {
    background: #0e1118;
    border: 1px solid #1a1f30;
    border-radius: 12px;
    padding: 1rem;
    margin-top: 0.5rem;
  }

  [data-testid="stSidebar"] label {
    color: #8a90aa !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.5px;
  }
  [data-testid="stSidebar"] .stNumberInput label,
  [data-testid="stSidebar"] .stSlider label {
    font-family: 'DM Mono', monospace !important;
  }

  .tag-train {
    background: #1a2f1a;
    color: #7acc7a;
    border: 1px solid #2a4a2a;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.68rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
  }
  .tag-val {
    background: #2a1f10;
    color: #d4944a;
    border: 1px solid #4a3018;
    border-radius: 6px;
    padding: 0.2rem 0.6rem;
    font-size: 0.68rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
  }

  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: #0d0f14; }
  ::-webkit-scrollbar-thumb { background: #2a3050; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


CSV_FILE = "stress_data.csv"
FEATURES = ["hours_slept", "caffeine_mg", "hours_study", "activity"]


def gendata(n: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    hours_slept       = np.random.normal(loc=7.0, scale=1.2, size=n).clip(4, 10)
    caffeine_mg       = np.random.normal(loc=180, scale=60,  size=n).clip(0, 500)
    hours_study       = np.random.normal(loc=6.0, scale=2.0, size=n).clip(0, 12)
    physical_activity = np.random.normal(loc=30,  scale=20,  size=n).clip(0, 120)

    stress = (
        - 0.7  * hours_slept
        + 0.004 * caffeine_mg
        + 0.35 * hours_study
        - 0.025 * physical_activity
        + np.random.normal(0, 0.5, n)
        + 5.5
    ).clip(1, 10)

    dates = [datetime.today() - timedelta(days=n - i) for i in range(n)]

    df = pd.DataFrame({
        "date":         [d.strftime("%Y-%m-%d") for d in dates],
        "hours_slept":  np.round(hours_slept, 2),
        "caffeine_mg":  np.round(caffeine_mg, 1),
        "hours_study":  np.round(hours_study, 2),
        "activity":     np.round(physical_activity, 1),
        "stress_score": np.round(stress, 2),
    })
    return df


def loaddataset():
    """Load existing CSV or regenerate if columns are missing/wrong."""
    REQUIRED_COLS = FEATURES + ["stress_score"]

    if not os.path.exists(CSV_FILE):
        df = gendata(50)
        df.to_csv(CSV_FILE, index=False)
        return df, True

    df = pd.read_csv(CSV_FILE)

   
    ALIASES = {
        "physical_activity_min": "activity",
        "physical_activity":     "activity",
        "activity_min":          "activity",
        "study_work_hours":      "hours_study",
        "work_hours":            "hours_study",
        "study_hours":           "hours_study",
        "sleep":                 "hours_slept",
        "hours_sleep":           "hours_slept",
        "caffeine":              "caffeine_mg",
        "caffeine_intake":       "caffeine_mg",
        "stress":                "stress_score",
    }
    df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns},
              inplace=True)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        df = gendata(50)
        df.to_csv(CSV_FILE, index=False)
        return df, True

    return df, False


def entry_save(hours_slept, caffeine_mg, hours_study, physical_activity, stress_score):
    df, _ = loaddataset()
    new_row = pd.DataFrame([{
        "date":         datetime.today().strftime("%Y-%m-%d"),
        "hours_slept":  hours_slept,
        "caffeine_mg":  caffeine_mg,
        "hours_study":  hours_study,
        "activity":     physical_activity,
        "stress_score": round(stress_score, 2),
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    return df


@st.cache_data(ttl=30)
def trainmodel(csv_path: str):
    df = pd.read_csv(csv_path)

  
    ALIASES = {
        "physical_activity_min": "activity",
        "physical_activity":     "activity",
        "activity_min":          "activity",
        "study_work_hours":      "hours_study",
        "work_hours":            "hours_study",
        "study_hours":           "hours_study",
        "sleep":                 "hours_slept",
        "hours_sleep":           "hours_slept",
        "caffeine":              "caffeine_mg",
        "caffeine_intake":       "caffeine_mg",
        "stress":                "stress_score",
    }
    df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns},
              inplace=True)

    
    df.to_csv(csv_path, index=False)

    X = df[FEATURES].values
    y = df["stress_score"].values

    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X, y, np.arange(len(df)), test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    model = LinearRegression()
    model.fit(X_train_s, y_train)

    y_pred_train = model.predict(X_train_s)
    y_pred_val   = model.predict(X_val_s)

    metrics = {
        "mse_train": mean_squared_error(y_train, y_pred_train),
        "mse_val":   mean_squared_error(y_val,   y_pred_val),
        "r2_train":  r2_score(y_train, y_pred_train),
        "r2_val":    r2_score(y_val,   y_pred_val),
        "n_train":   len(X_train),
        "n_val":     len(X_val),
    }
    return model, scaler, metrics, df, idx_train, idx_val




with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.2rem">
      <div style="font-family:'Syne',sans-serif;font-size:1.25rem;font-weight:800;
                  color:#c8f0a0;letter-spacing:-0.5px;">🧠 Daily Inputs</div>
      <div style="font-size:0.68rem;color:#4a5070;letter-spacing:2px;
                  text-transform:uppercase;margin-top:0.15rem;">
        Enter today's habits</div>
    </div>
    """, unsafe_allow_html=True)   # FIX 2 applied

    hours_slept = st.slider(
        "😴 Hours Slept",
        min_value=4.0, max_value=10.0, value=7.0, step=0.5,
        help="Total hours of sleep last night"
    )
    caffeine_mg = st.number_input(
        "☕ Caffeine Intake (mg)",
        min_value=0, max_value=600, value=180, step=10,
        help="Total caffeine consumed today (espresso ≈ 63mg, drip coffee ≈ 95mg)"
    )
    hours_study = st.slider(
        "📚 Study / Work Hours",
        min_value=0.0, max_value=12.0, value=6.0, step=0.5,
        help="Focused study or work hours today"
    )
    physical_activity = st.slider(
        "🏃 Physical Activity (min)",
        min_value=0, max_value=120, value=30, step=5,
        help="Total active exercise or movement in minutes"
    )

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    log_today = st.button("💾 Save Today's Entry", use_container_width=True)

    st.markdown("""
    <div style="margin-top:1.5rem;font-size:0.65rem;color:#3a4060;
                line-height:1.8;letter-spacing:0.3px;">
      Data stored locally in <code>stress_data.csv</code>.<br>
      Model retrains on every app load.<br><br>
      <span style="color:#4a6090;">CO3</span> — Normal Distribution seed data<br>
      <span style="color:#4a6090;">CO4</span> — Linear Regression · MSE · R²
    </div>
    """, unsafe_allow_html=True)

raw_df, was_generated = loaddataset()

if was_generated:
    st.toast("✨ 50 synthetic seed days generated via Normal Distribution (CO3)", icon="🌱")

model, scaler, metrics, df, idx_train, idx_val = trainmodel(CSV_FILE)



input_data  = np.array([[hours_slept, caffeine_mg, hours_study, physical_activity]])
user_scaled = scaler.transform(input_data)
stress_pred = float(model.predict(user_scaled)[0])
stress_pred = round(np.clip(stress_pred, 1.0, 10.0), 2)


def stress_color(score):
    if score <= 3.5: return "#7acc7a", "LOW — You're doing well!"
    if score <= 6.0: return "#f0c060", "MODERATE — Stay mindful."
    if score <= 8.0: return "#e0884a", "HIGH — Take a break."
    return "#e05555", "CRITICAL — Rest & recover!"

s_color, s_label = stress_color(stress_pred)

if log_today:
    df = entry_save(hours_slept, caffeine_mg, hours_study, physical_activity, stress_pred)
    st.cache_data.clear()
    st.toast(f"Entry saved! Predicted stress: {stress_pred}", icon="✅")
    st.rerun()




st.markdown("""
<div style="margin-bottom:1.5rem">
  <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
              color:#e8eaf0;letter-spacing:-1px;line-height:1.1;">
    Personal AI Stress Predictor
  </div>
  <div style="font-size:0.72rem;color:#3a4565;letter-spacing:2.5px;
              text-transform:uppercase;margin-top:0.3rem;">
    Linear Regression · Scikit-Learn · CO3 / CO4
  </div>
</div>
""", unsafe_allow_html=True)

col_pred, col_metrics = st.columns([1, 2], gap="large")

with col_pred:
    st.markdown('<div class="section-header">Today\'s Prediction</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class="stress-badge">
      <div class="stress-label">Stress Score</div>
      <div class="stress-score-number" style="color:{s_color};">{stress_pred}</div>
      <div style="font-size:0.68rem;letter-spacing:2px;color:{s_color};
                  margin-top:0.4rem;opacity:0.85;">{s_label}</div>
      <div style="margin-top:1rem;display:grid;grid-template-columns:1fr 1fr;gap:0.35rem;">
        <span class="pill">😴 {hours_slept}h sleep</span>
        <span class="pill">☕ {caffeine_mg}mg</span>
        <span class="pill">📚 {hours_study}h work</span>
        <span class="pill">🏃 {physical_activity}min</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    st.markdown('<div class="section-header">Model Performance (CO4)</div>',
                unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("MSE — Training Set",  f"{metrics['mse_train']:.4f}")
        st.metric("R² Score — Training", f"{metrics['r2_train']:.4f}")
        st.markdown(f'<span class="tag-train">Training Set · {metrics["n_train"]} rows</span>',
                    unsafe_allow_html=True)
    with m2:
        st.metric("MSE — Validation Set",  f"{metrics['mse_val']:.4f}")
        st.metric("R² Score — Validation", f"{metrics['r2_val']:.4f}")
        st.markdown(f'<span class="tag-val">Validation Set · {metrics["n_val"]} rows</span>',
                    unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:0.8rem;font-size:0.7rem;color:#3a4565;line-height:1.9">
      <b style="color:#5a6580;">MSE</b> = Mean Squared Error — lower is better.<br>
      <b style="color:#5a6580;">R²</b>  = Coefficient of Determination — closer to 1.0 is better.<br>
      Dataset split: <span style="color:#7acc7a;">80% Train</span> /
      <span style="color:#d4944a;">20% Validate</span> (random_state=42).
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

col_trend, col_fi = st.columns([3, 2], gap="large")

DARK_BG  = "#0e1118"
GRID_CLR = "#1a1f2e"
TEXT_CLR = "#8090b0"

with col_trend:
    st.markdown('<div class="section-header">Stress Trend Over Time</div>',
                unsafe_allow_html=True)

    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    plot_df = plot_df.sort_values("date").reset_index(drop=True)

    n_total   = len(plot_df)
    split_idx = int(n_total * 0.8)

    fig_trend, ax_trend = plt.subplots(figsize=(9, 3.6), facecolor=DARK_BG)
    ax_trend.set_facecolor(DARK_BG)

    plot_df["rolling"] = plot_df["stress_score"].rolling(5, min_periods=1).mean()

    train_x = plot_df.index[:split_idx + 1]
    val_x   = plot_df.index[split_idx:]

    ax_trend.fill_between(train_x, 0, 11, color="#7acc7a", alpha=0.04)
    ax_trend.fill_between(val_x,   0, 11, color="#d4944a", alpha=0.07)

    ax_trend.scatter(plot_df.index[:split_idx], plot_df["stress_score"][:split_idx],
                     color="#4a8a4a", s=18, zorder=4, alpha=0.7)
    ax_trend.scatter(plot_df.index[split_idx:], plot_df["stress_score"][split_idx:],
                     color="#c07030", s=18, zorder=4, alpha=0.9)

    ax_trend.plot(plot_df.index[:split_idx + 1], plot_df["stress_score"][:split_idx + 1],
                  color="#7acc7a", lw=1.2, alpha=0.5)
    ax_trend.plot(plot_df.index[split_idx:], plot_df["stress_score"][split_idx:],
                  color="#d4944a", lw=1.5, alpha=0.7)

    ax_trend.plot(plot_df.index, plot_df["rolling"],
                  color="#c8f0a0", lw=2.0, alpha=0.9, ls="--", label="5-day rolling avg")

    ax_trend.axvline(split_idx, color="#2a3a5e", lw=1.5, ls=":")
    ax_trend.axhline(stress_pred, color=s_color, lw=1.0, ls=":", alpha=0.6,
                     label=f"Today's prediction ({stress_pred})")

    train_patch = mpatches.Patch(color="#7acc7a", label="Training Set (80%)", alpha=0.7)
    val_patch   = mpatches.Patch(color="#d4944a", label="Validation Set (20%)", alpha=0.8)
    ax_trend.legend(handles=[train_patch, val_patch],
                    loc="upper left", framealpha=0, fontsize=7, labelcolor="#8090b0")

    ax_trend.set_ylim(0.5, 10.5)
    ax_trend.set_xlim(-0.5, n_total - 0.5)
    ax_trend.set_xlabel("Day index", fontsize=7, color=TEXT_CLR)
    ax_trend.set_ylabel("Stress Score", fontsize=7, color=TEXT_CLR)
    ax_trend.tick_params(colors=TEXT_CLR, labelsize=6.5)
    for spine in ax_trend.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax_trend.yaxis.grid(True, color=GRID_CLR, lw=0.7)
    ax_trend.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    st.pyplot(fig_trend, use_container_width=True)
    plt.close(fig_trend)


with col_fi:
    st.markdown('<div class="section-header">Feature Importance (Linear Coefficients)</div>',
                unsafe_allow_html=True)

    feature_labels = ["Hours\nSlept", "Caffeine\n(mg)", "Work/Study\nHours", "Physical\nActivity"]
    coefficients   = model.coef_
    bar_colors     = ["#e05555" if c > 0 else "#7acc7a" for c in coefficients]

    fig_fi, ax_fi = plt.subplots(figsize=(7, 4.8), facecolor=DARK_BG)
    ax_fi.set_facecolor(DARK_BG)

    bars = ax_fi.barh(feature_labels, coefficients,
                      color=bar_colors, edgecolor="none", height=0.58)

    max_abs = max(abs(coefficients)) if max(abs(coefficients)) > 0 else 1
    for bar, val in zip(bars, coefficients):
        bar_w  = bar.get_width()
        bar_cx = bar.get_x() + bar_w / 2
        bar_cy = bar.get_y() + bar.get_height() / 2
        x_pos  = bar_cx if abs(bar_w) / max_abs >= 0.15 else val * 0.5
        ax_fi.text(x_pos, bar_cy, f"{val:+.3f}",
                   va="center", ha="center",
                   color="#0d0f14", fontsize=9,
                   fontweight="bold", fontfamily="monospace")

    ax_fi.axvline(0, color="#2a3050", lw=1.2)
    ax_fi.set_xlabel("Standardised coefficient (impact on stress score)",
                     fontsize=6.5, color=TEXT_CLR)
    ax_fi.tick_params(colors=TEXT_CLR, labelsize=7.5)
    ax_fi.invert_yaxis()
    for spine in ax_fi.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax_fi.xaxis.grid(True, color=GRID_CLR, lw=0.6)
    ax_fi.set_axisbelow(True)

    pos_patch = mpatches.Patch(color="#e05555", label="Raises stress ↑")
    neg_patch = mpatches.Patch(color="#7acc7a", label="Reduces stress ↓")
    ax_fi.legend(handles=[pos_patch, neg_patch], loc="lower right",
                 framealpha=0, fontsize=6.5, labelcolor="#8090b0")

    plt.tight_layout(pad=0.6)
    st.pyplot(fig_fi, use_container_width=True)
    plt.close(fig_fi)

st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

with st.expander("📋 Dataset Preview — stress_data.csv", expanded=False):
    st.markdown(
        f'<span class="tag-train">Total Rows: {len(df)}</span>&nbsp;'
        f'<span class="tag-val">Features: {len(FEATURES)}</span>',
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

    def highlight_stress(val):
        if val <= 3.5:   return "color: #7acc7a"
        elif val <= 6.0: return "color: #f0c060"
        elif val <= 8.0: return "color: #e0884a"
        return "color: #e05555"

    styled = (
        df.tail(20)
          .style
          .applymap(highlight_stress, subset=["stress_score"])
          .format({
              "hours_slept":  "{:.1f}",
              "caffeine_mg":  "{:.0f}",
              "hours_study":  "{:.1f}",
              "activity":     "{:.0f}",
              "stress_score": "{:.2f}",
          })
          .set_properties(**{
              "background-color": "#0d0f14",
              "color": "#c0c8e0",
              "font-size": "0.74rem",
              "font-family": "DM Mono, monospace",
          })
    )
    st.dataframe(styled, use_container_width=True, height=320)

st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;
            font-size:0.62rem;color:#2a3050;letter-spacing:2px;
            text-transform:uppercase;font-family:'DM Mono',monospace;">
  Personal AI Stress Predictor · Linear Regression · CO3 Normal Distribution · CO4 Scikit-Learn
</div>
""", unsafe_allow_html=True)
