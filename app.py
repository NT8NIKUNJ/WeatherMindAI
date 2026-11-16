import os
import joblib
import numpy as np
import pandas as pd
import textwrap
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.models import load_model
from src.live_weather import get_live_weather

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WeatherMind AI",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATA_FILE = os.path.join(
    DATA_DIR,
    "weather_model_ready.csv"
)

MODEL_COMPARISON_FILE = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

RF_IMPORTANCE_FILE = os.path.join(
    MODELS_DIR,
    "random_forest_feature_importance.csv"
)

XGB_IMPORTANCE_FILE = os.path.join(
    MODELS_DIR,
    "xgboost_feature_importance.csv"
)

LSTM_HISTORY_FILE = os.path.join(
    MODELS_DIR,
    "lstm_training_history.csv"
)



# ============================================================
# GLOBAL VISUAL THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --wm-blue: #2563eb;
        --wm-cyan: #06b6d4;
        --wm-navy: #0f172a;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 0%, rgba(37,99,235,.07), transparent 28%),
            radial-gradient(circle at 95% 5%, rgba(6,182,212,.06), transparent 24%);
    }

    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1450px;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    h1 {
        font-weight: 800 !important;
    }

    h2, h3 {
        font-weight: 700 !important;
    }

    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.20);
        padding: 18px 20px;
        border-radius: 16px;
        box-shadow: 0 5px 18px rgba(15,23,42,.045);
    }

    [data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    [data-testid="stMetricValue"] {
        font-weight: 800;
    }

    div[data-testid="stVerticalBlock"] > div:has(> [data-testid="stPlotlyChart"]) {
        border-radius: 16px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128,128,128,.18);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.35rem;
    }

    [data-testid="stSidebar"] .stRadio label {
        padding-top: 3px;
        padding-bottom: 3px;
        font-weight: 550;
    }

    .wm-hero {
        padding: 30px 34px;
        border-radius: 22px;
        margin: 4px 0 24px 0;
        color: white;
        background: linear-gradient(120deg, #0b1b38 0%, #174b83 58%, #087e9b 100%);
        box-shadow: 0 12px 32px rgba(15, 55, 100, .18);
    }

    .wm-hero h1 {
        color: white !important;
        font-size: clamp(2rem, 4vw, 3rem);
        margin: 0 0 8px 0;
    }

    .wm-hero p {
        color: #e2f2ff !important;
        font-size: 1.04rem;
        margin: 0;
        line-height: 1.65;
    }

    .wm-eyebrow {
        color: #bae6fd;
        text-transform: uppercase;
        letter-spacing: .13em;
        font-size: .76rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .wm-section-note {
        color: var(--text-color);
        opacity: .72;
        margin-top: -8px;
        margin-bottom: 18px;
    }

    .wm-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 132px;
        box-shadow: 0 5px 18px rgba(15,23,42,.035);
    }

    .wm-card h3 {
        font-size: 1.03rem;
        margin: 0 0 8px 0;
    }

    .wm-card p {
        font-size: .91rem;
        opacity: .76;
        line-height: 1.55;
        margin: 0;
    }

    .wm-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 999px;
        background: rgba(6,182,212,.13);
        color: #0891b2;
        font-size: .78rem;
        font-weight: 750;
        margin-bottom: 10px;
    }

    /* Existing About Project page components */
    .hero {
        padding: 30px 34px;
        border-radius: 20px;
        margin-bottom: 24px;
        background: linear-gradient(120deg, #0b1b38, #174b83, #087e9b);
        color: white;
    }
    .hero h1 { color: white !important; font-size: 2.5rem; }
    .hero p { color: #e2f2ff !important; line-height: 1.7; }
    .section-heading {
        font-size: 1.45rem;
        font-weight: 800;
        margin-top: 28px;
        margin-bottom: 14px;
    }
    .feature-card, .workflow-card, .footer-box {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,.20);
        border-radius: 15px;
        padding: 19px;
        margin-bottom: 12px;
        box-shadow: 0 5px 18px rgba(15,23,42,.035);
    }
    .feature-card { min-height: 145px; }
    .feature-card h3, .workflow-card h3 { font-size: 1rem; }
    .feature-card p, .workflow-card p { opacity: .76; line-height: 1.55; }
    .workflow-card { text-align: center; min-height: 125px; }
    .footer-box { text-align: center; margin-top: 30px; padding: 26px; }

    @media (max-width: 700px) {
        .wm-hero { padding: 23px; }
        .block-container { padding-top: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_weather_data():

    if not os.path.exists(DATA_FILE):
        return None

    df = pd.read_csv(DATA_FILE)

    df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_data
def load_model_comparison():

    if not os.path.exists(MODEL_COMPARISON_FILE):
        return None

    return pd.read_csv(MODEL_COMPARISON_FILE)


@st.cache_data
def load_rf_importance():

    if not os.path.exists(RF_IMPORTANCE_FILE):
        return None

    return pd.read_csv(RF_IMPORTANCE_FILE)


@st.cache_data
def load_xgb_importance():

    if not os.path.exists(XGB_IMPORTANCE_FILE):
        return None

    return pd.read_csv(XGB_IMPORTANCE_FILE)


@st.cache_data
def load_lstm_history():

    if not os.path.exists(LSTM_HISTORY_FILE):
        return None

    return pd.read_csv(LSTM_HISTORY_FILE)


weather_df = load_weather_data()
comparison_df = load_model_comparison()
rf_importance = load_rf_importance()
xgb_importance = load_xgb_importance()
lstm_history = load_lstm_history()


# ============================================================
# LOAD TRAINED MODELS
# ============================================================

@st.cache_resource
def load_trained_models():

    models = {}

    linear_path = os.path.join(
        MODELS_DIR,
        "linear_regression.pkl"
    )

    rf_path = os.path.join(
        MODELS_DIR,
        "random_forest.pkl"
    )

    xgb_path = os.path.join(
        MODELS_DIR,
        "xgboost_model.pkl"
    )

    lstm_path = os.path.join(
        MODELS_DIR,
        "lstm_model.keras"
    )

    if os.path.exists(linear_path):
        models["Linear Regression"] = joblib.load(
            linear_path
        )

    if os.path.exists(rf_path):
        models["Random Forest"] = joblib.load(
            rf_path
        )

    if os.path.exists(xgb_path):
        models["XGBoost"] = joblib.load(
            xgb_path
        )

    if os.path.exists(lstm_path):
        models["LSTM"] = load_model(
            lstm_path
        )

    return models


models = load_trained_models()



# ============================================================
# APPLICATION HEADER
# ============================================================

st.markdown(
    """
    <div class="wm-hero">
        <div class="wm-eyebrow">AI-powered weather intelligence</div>
        <h1>🌦️ WeatherMind AI</h1>
        <p>
            Intelligent next-day temperature prediction, live weather
            insights and configurable weather-risk screening.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌦️ WeatherMind AI")

st.sidebar.markdown(
    """
    ### Navigation

    Use the menu below to explore the forecasting system.
    """
)

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Dashboard",
        "🔮 Prediction",
        "📊 Model Comparison",
        "🔍 Feature Analysis",
        "📈 Historical Weather",
        "🌦️ Live Forecast",
        "⚠️ Weather Risk Intelligence",
        "🧪 AI Forecast Comparison Lab",
        "🇮🇳 India Weather Explorer",
        "🧠 About Project"
    ]
)



# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("Weather Intelligence Dashboard")
    st.markdown(
        '<div class="wm-section-note">A consolidated view of historical '
        'weather patterns, dataset coverage and model performance.</div>',
        unsafe_allow_html=True
    )

    if weather_df is None:
        st.error("Weather dataset not found. Please check the data folder.")
        st.stop()

    # Top-level dataset KPIs
    first_col, second_col, third_col, fourth_col = st.columns(4)

    with first_col:
        st.metric("Historical records", f"{len(weather_df):,}")

    with second_col:
        st.metric(
            "Mean temperature",
            f"{weather_df['temperature_mean'].mean():.1f} °C"
        )

    with third_col:
        st.metric(
            "Mean humidity",
            f"{weather_df['humidity_mean'].mean():.1f}%"
        )

    with fourth_col:
        model_count = len(models)
        st.metric("Loaded AI models", f"{model_count}")

    st.divider()

    # Dataset coverage and best model
    left, right = st.columns([1.15, 1])

    with left:
        st.subheader("📅 Dataset coverage")
        st.markdown(
            f"""
            <div class="wm-card">
                <span class="wm-pill">Delhi · India</span>
                <h3>Historical weather observations</h3>
                <p>
                    <b>From:</b> {weather_df['date'].min().strftime('%d %b %Y')}<br>
                    <b>To:</b> {weather_df['date'].max().strftime('%d %b %Y')}<br>
                    <b>Records:</b> {len(weather_df):,}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with right:
        st.subheader("🏆 Best evaluated model")
        if comparison_df is not None and not comparison_df.empty and "MAE" in comparison_df.columns:
            best_row = comparison_df.loc[comparison_df["MAE"].idxmin()]
            best_name = best_row.get("Model", "Best model")
            best_mae = float(best_row["MAE"])
            st.markdown(
                f"""
                <div class="wm-card">
                    <span class="wm-pill">Lowest test MAE</span>
                    <h3>{best_name}</h3>
                    <p>
                        Mean Absolute Error<br>
                        <strong style="font-size:1.65rem">{best_mae:.4f} °C</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Model comparison results are not available yet.")

    st.divider()

    # Historical temperature trend
    st.subheader("🌡️ Historical temperature trend")
    st.markdown(
        '<div class="wm-section-note">Daily mean temperature across the available historical dataset.</div>',
        unsafe_allow_html=True
    )

    trend_fig = px.line(
        weather_df,
        x="date",
        y="temperature_mean",
        labels={
            "date": "Date",
            "temperature_mean": "Mean temperature (°C)"
        }
    )
    trend_fig.update_traces(line=dict(width=2, color="#0891b2"))
    trend_fig.update_layout(
        template="plotly_white",
        height=390,
        margin=dict(l=15, r=15, t=25, b=15),
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title="Temperature (°C)",
        font=dict(family="Arial", size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(trend_fig, use_container_width=True)

    # Variable explorer
    st.subheader("📊 Explore weather variables")
    variable_options = {
        "Mean temperature (°C)": "temperature_mean",
        "Maximum temperature (°C)": "temperature_max",
        "Minimum temperature (°C)": "temperature_min",
        "Mean humidity (%)": "humidity_mean",
        "Pressure": "pressure",
        "Mean wind speed": "wind_speed_mean",
        "Cloud cover": "cloud_cover",
        "Rainfall": "rainfall",
        "Solar radiation": "solar_radiation"
    }

    selected_label = st.selectbox(
        "Choose a weather variable",
        list(variable_options.keys())
    )
    selected_variable = variable_options[selected_label]

    variable_fig = px.area(
        weather_df,
        x="date",
        y=selected_variable,
        labels={
            "date": "Date",
            selected_variable: selected_label
        }
    )
    variable_fig.update_traces(
        line=dict(width=1.8, color="#2563eb"),
        fillcolor="rgba(37,99,235,0.12)"
    )
    variable_fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=15, r=15, t=20, b=15),
        xaxis_title=None,
        yaxis_title=selected_label,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(variable_fig, use_container_width=True)

    # Model comparison preview
    if comparison_df is not None and not comparison_df.empty:
        st.subheader("🤖 Model performance snapshot")
        required_metrics = {"Model", "MAE", "RMSE", "R2"}
        if required_metrics.issubset(comparison_df.columns):
            preview = comparison_df.copy()
            preview = preview.sort_values("MAE", ascending=True)
            st.dataframe(
                preview,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# PREDICTION
# ============================================================

elif page == "🔮 Prediction":

    st.header("🔮 Next-Day Temperature Prediction")

    if weather_df is None:
        st.error("Weather dataset not found.")
        st.stop()

    st.write(
        """
        Select a date from the available historical dataset.
        WeatherMind AI will use the features for that date
        to predict the **next day's mean temperature**.
        """
    )

    prediction_dates = weather_df["date"].iloc[:-1]

    selected_date = st.date_input(
        "Select Date",
        value=prediction_dates.iloc[-1].date(),
        min_value=prediction_dates.min().date(),
        max_value=prediction_dates.max().date()
    )

    selected_timestamp = pd.Timestamp(
        selected_date
    )

    row = weather_df[
        weather_df["date"] == selected_timestamp
    ]

    if row.empty:

        st.warning(
            "No data available for the selected date."
        )

    else:

        row = row.iloc[0]

        
        # Get the correct numerical features for prediction
        feature_columns = [
            column
            for column in weather_df.columns
            if column not in ["date", "target_temperature"]
            and pd.api.types.is_numeric_dtype(weather_df[column])
        ]

        # Create input using only numerical features
        X_input = pd.DataFrame(
            [row[feature_columns].values],
            columns=feature_columns
        )


        st.subheader("🌤️ Selected Weather Conditions")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Temperature",
                f"{row['temperature_mean']:.1f} °C"
            )

        with c2:
            st.metric(
                "Humidity",
                f"{row['humidity_mean']:.0f}%"
            )

        with c3:
            st.metric(
                "Pressure",
                f"{row['pressure']:.0f}"
            )

        with c4:
            st.metric(
                "Wind Speed",
                f"{row['wind_speed_mean']:.1f} km/h"
            )

        st.divider()

        st.subheader("🤖 Model Predictions")

        model_names = [
            "Linear Regression",
            "Random Forest",
            "XGBoost"
        ]

        prediction_results = []

        for model_name in model_names:
            if model_name in models:
                model = models[model_name]
                prediction = model.predict(X_input)[0]

                prediction_results.append(
                    (model_name, float(prediction))
                )

        if prediction_results:
            cols = st.columns(3)

            for col, (name, value) in zip(cols, prediction_results):
                with col:
                    st.metric(
                        label=name,
                        value=f"{value:.2f} °C"
                    )

            best_prediction = prediction_results[0][1]

            st.success(
                f"🌡️ WeatherMind AI estimate: "
                f"**{best_prediction:.2f} °C** for the next day."
            )
        else:
            st.warning("No prediction models are available.")

# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "📊 Model Comparison":

    st.header("📊 Model Performance Comparison")

    if comparison_df is None:

        st.error(
            "Model comparison results not found."
        )

        st.stop()

    st.write(
        """
        Four forecasting approaches were evaluated
        using the same chronological train/test setup.
        """
    )

    display_df = comparison_df.copy()

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    st.subheader("📉 Mean Absolute Error")

    fig_mae = px.bar(
        comparison_df,
        x="Model",
        y="MAE",
        title="Model Comparison - MAE",
        text_auto=".4f"
    )

    st.plotly_chart(
        fig_mae,
        use_container_width=True
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    st.subheader("📉 Root Mean Squared Error")

    fig_rmse = px.bar(
        comparison_df,
        x="Model",
        y="RMSE",
        title="Model Comparison - RMSE",
        text_auto=".4f"
    )

    st.plotly_chart(
        fig_rmse,
        use_container_width=True
    )

    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    st.subheader("📈 R² Score")

    fig_r2 = px.bar(
        comparison_df,
        x="Model",
        y="R2",
        title="Model Comparison - R²",
        text_auto=".4f"
    )

    st.plotly_chart(
        fig_r2,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_model = comparison_df.loc[
        comparison_df["MAE"].idxmin(),
        "Model"
    ]

    best_mae = comparison_df["MAE"].min()

    st.success(
        f"🏆 Best model on the test set: "
        f"**{best_model}** "
        f"with MAE = **{best_mae:.4f} °C**."
    )


# ============================================================
# FEATURE ANALYSIS
# ============================================================

elif page == "🔍 Feature Analysis":

    st.header("🔍 Feature Analysis & Model Transparency")

    st.write(
        """
        Feature importance helps us understand which
        weather variables contributed most strongly to
        tree-based model predictions.
        """
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    if rf_importance is not None:

        st.subheader("🌲 Random Forest Feature Importance")

        rf_plot = rf_importance.sort_values(
            "importance",
            ascending=False
        ).head(15)

        rf_plot = rf_plot.sort_values(
            "importance",
            ascending=True
        )

        fig_rf = px.bar(
            rf_plot,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Random Forest Features"
        )

        st.plotly_chart(
            fig_rf,
            use_container_width=True
        )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    if xgb_importance is not None:

        st.subheader("⚡ XGBoost Feature Importance")

        xgb_plot = xgb_importance.sort_values(
            "importance",
            ascending=False
        ).head(15)

        xgb_plot = xgb_plot.sort_values(
            "importance",
            ascending=True
        )

        fig_xgb = px.bar(
            xgb_plot,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 XGBoost Features"
        )

        st.plotly_chart(
            fig_xgb,
            use_container_width=True
        )

    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    if lstm_history is not None:

        st.subheader("🧠 LSTM Training Analysis")

        lstm_history = lstm_history.copy()

        lstm_history["epoch"] = range(
            1,
            len(lstm_history) + 1
        )

        fig_lstm = px.line(
            lstm_history,
            x="epoch",
            y=["loss", "val_loss"],
            title="LSTM Training vs Validation Loss",
            labels={
                "epoch": "Epoch",
                "value": "Loss",
                "variable": "Metric"
            }
        )

        st.plotly_chart(
            fig_lstm,
            use_container_width=True
        )


# ============================================================
# HISTORICAL WEATHER
# ============================================================

elif page == "📈 Historical Weather":

    st.header("📈 Historical Weather Data")

    if weather_df is None:

        st.error("Weather dataset not found.")
        st.stop()

    min_date = weather_df["date"].min().date()
    max_date = weather_df["date"].max().date()

    date_range = st.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:

        start_date = pd.Timestamp(
            date_range[0]
        )

        end_date = pd.Timestamp(
            date_range[1]
        )

        filtered_df = weather_df[
            (weather_df["date"] >= start_date)
            &
            (weather_df["date"] <= end_date)
        ]

        st.write(
            f"Showing **{len(filtered_df):,} records**"
        )

        display_columns = [
            "date",
            "temperature_mean",
            "temperature_max",
            "temperature_min",
            "humidity_mean",
            "pressure",
            "wind_speed_mean",
            "rainfall",
            "cloud_cover"
        ]

        st.dataframe(
            filtered_df[display_columns],
            use_container_width=True,
            hide_index=True
        )
        
        st.subheader("Temperature Trends")

        temperature_chart = filtered_df.set_index("date")[
            [
                "temperature_mean",
                "temperature_max",
                "temperature_min"
            ]
        ]

        st.line_chart(temperature_chart)

        st.subheader("Humidity Trends")

        st.line_chart(
            filtered_df.set_index("date")[["humidity_mean"]]
        )

        st.subheader("Rainfall Trends")

        st.bar_chart(
            filtered_df.set_index("date")[["rainfall"]]
        )


elif page == "🌦️ Live Forecast":

    st.header("🌦️ Live Weather Forecast")
    cities = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Chennai": (13.0827, 80.2707),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873),
    "Lucknow": (26.8467, 80.9462),
    }

    selected_city = st.selectbox(
    "📍 Select an Indian city",
    list(cities.keys())
    )

    latitude, longitude = cities[selected_city]

    st.write(
        f"Live weather conditions and a 7-day forecast "
        f"for **{selected_city}, India**."
    )

    try:
        with st.spinner("Fetching live weather data..."):
            weather = get_live_weather(latitude, longitude)

        current = weather["current"]
        daily = weather["daily"]
        hourly = weather["hourly"]

        st.caption(
            f"Last updated: {current['time']} "
            f"| Time zone: {weather['timezone']}"
        )

        # Current weather cards
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "🌡️ Temperature",
            f"{current['temperature_2m']} °C"
        )
        col2.metric(
            "🤗 Feels Like",
            f"{current['apparent_temperature']} °C"
        )
        col3.metric(
            "💧 Humidity",
            f"{current['relative_humidity_2m']}%"
        )
        col4.metric(
            "💨 Wind Speed",
            f"{current['wind_speed_10m']} km/h"
        )

        st.subheader("📅 7-Day Forecast")

        daily_display = daily.copy()
        daily_display["time"] = pd.to_datetime(
            daily_display["time"]
        ).dt.strftime("%d %b %Y")

        st.dataframe(
            daily_display.rename(columns={
                "time": "Date",
                "temperature_2m_max": "Max Temperature (°C)",
                "temperature_2m_min": "Min Temperature (°C)",
                "precipitation_probability_max": "Rain Chance (%)",
                "precipitation_sum": "Precipitation (mm)",
                "wind_speed_10m_max": "Max Wind (km/h)",
                "weather_code": "Weather Code"
            }),
            use_container_width=True,
            hide_index=True
        )

        st.subheader("🌡️ Daily Temperature Forecast")

        daily_chart = daily.copy()
        daily_chart["time"] = pd.to_datetime(daily_chart["time"])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_chart["time"],
            y=daily_chart["temperature_2m_max"],
            mode="lines+markers",
            name="Maximum Temperature"
        ))

        fig.add_trace(go.Scatter(
            x=daily_chart["time"],
            y=daily_chart["temperature_2m_min"],
            mode="lines+markers",
            name="Minimum Temperature"
        ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Temperature (°C)",
            hovermode="x unified"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("🕒 Hourly Forecast")

        hours = hourly.copy()
        hours["time"] = pd.to_datetime(hours["time"])
        hours = hours.head(24)

        hourly_chart = px.line(
            hours,
            x="time",
            y="temperature_2m",
            markers=True,
            title="Temperature for the Next 24 Hours",
            labels={
                "time": "Time",
                "temperature_2m": "Temperature (°C)"
            }
        )

        st.plotly_chart(hourly_chart, use_container_width=True)

        st.caption(
            "Forecast data provided by Open-Meteo. "
            "Forecasts may change as new weather data becomes available."
        )

    except Exception as e:
        st.error(
            "Unable to fetch live weather right now. "
            "Please check your internet connection and try again."
        )
        st.caption(f"Technical details: {e}")


elif page == "⚠️ Weather Risk Intelligence":

    st.header("⚠️ Smart Weather Risk Intelligence")
    st.write(
        "Analyze the upcoming forecast to identify potential "
        "heat, rainfall, and high-wind risks."
    )

    try:
        with st.spinner("Analyzing the 7-day forecast..."):
            cities = {
                "Delhi": (28.6139, 77.2090),
                "Mumbai": (19.0760, 72.8777),
                "Chennai": (13.0827, 80.2708),
                "Bengaluru": (12.9716, 77.5946),
                "Kolkata": (22.5726, 88.3639),
                "Hyderabad": (17.3850, 78.4867),
                "Jaipur": (26.9124, 75.7873),
                "Lucknow": (26.8467, 80.9462),
            }

            selected_risk_city = st.selectbox(
                "📍 Select city for risk analysis",
                list(cities.keys()),
                key="risk_city"
            )

            latitude, longitude = cities[selected_risk_city]
            weather = get_live_weather(latitude, longitude)

            st.write(f"Analyzing the forecast for **{selected_risk_city}**.")

        daily = weather["daily"]

        # Explainable, configurable screening thresholds
        heat_threshold = st.slider(
            "High-temperature threshold (°C)",
            min_value=30,
            max_value=45,
            value=40
        )

        rain_threshold = st.slider(
            "Daily precipitation threshold (mm)",
            min_value=5,
            max_value=100,
            value=25
        )

        wind_threshold = st.slider(
            "Maximum wind speed threshold (km/h)",
            min_value=10,
            max_value=100,
            value=40
        )

        st.caption(
            "These are configurable project screening thresholds, "
            "not official meteorological warning criteria."
        )

        risks = []

        for _, day in daily.iterrows():
            date = str(day["time"])

            if day["temperature_2m_max"] >= heat_threshold:
                risks.append({
                    "Date": date,
                    "Risk Type": "High temperature",
                    "Forecast Value": (
                        f'{day["temperature_2m_max"]:.1f} °C'
                    ),
                    "Threshold": f"{heat_threshold} °C",
                    "Explanation": (
                        "Maximum forecast temperature meets or "
                        "exceeds the selected threshold."
                    )
                })

            if day["precipitation_sum"] >= rain_threshold:
                risks.append({
                    "Date": date,
                    "Risk Type": "Heavy precipitation screening",
                    "Forecast Value": (
                        f'{day["precipitation_sum"]:.1f} mm'
                    ),
                    "Threshold": f"{rain_threshold} mm",
                    "Explanation": (
                        "Forecast daily precipitation meets or "
                        "exceeds the selected threshold."
                    )
                })

            if day["wind_speed_10m_max"] >= wind_threshold:
                risks.append({
                    "Date": date,
                    "Risk Type": "High wind",
                    "Forecast Value": (
                        f'{day["wind_speed_10m_max"]:.1f} km/h'
                    ),
                    "Threshold": f"{wind_threshold} km/h",
                    "Explanation": (
                        "Maximum forecast wind speed meets or "
                        "exceeds the selected threshold."
                    )
                })

        st.subheader("📋 Risk Analysis Results")

        if risks:
            st.warning(
                f"{len(risks)} potential risk condition(s) "
                "detected in the forecast."
            )

            risk_df = pd.DataFrame(risks)
            st.dataframe(
                risk_df,
                use_container_width=True,
                hide_index=True
            )

            st.subheader("📊 Risk Conditions by Type")

            counts = risk_df["Risk Type"].value_counts().reset_index()
            counts.columns = ["Risk Type", "Count"]

            fig = px.bar(
                counts,
                x="Risk Type",
                y="Count",
                color="Risk Type",
                title="Detected Forecast Conditions"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.success(
                "No forecast values crossed your selected "
                "risk-screening thresholds."
            )

        st.caption(
            "Source: Open-Meteo forecast API. Results are automated "
            "screening indicators, not official warnings or guarantees "
            "of safety. Check local meteorological advisories."
        )

    except Exception as e:
        st.error(
            "Risk analysis is temporarily unavailable. "
            "Check your internet connection and try again."
        )
        st.caption(f"Technical details: {e}")


elif page == "🧪 AI Forecast Comparison Lab":

    st.header("🧪 AI Forecast Comparison Lab")

    st.write(
        "Compare your trained machine-learning models with "
        "the live weather forecast provider."
    )

    st.info(
        "This lab compares predictions for the same target date. "
        "Historical model metrics and live forecast comparisons "
        "are different experiments."
    )

    # Select a city
    cities = {
        "Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Chennai": (13.0827, 80.2708),
        "Bengaluru": (12.9716, 77.5946),
        "Kolkata": (22.5726, 88.3639),
        "Hyderabad": (17.3850, 78.4867),
        "Jaipur": (26.9124, 75.7873),
        "Lucknow": (26.8467, 80.9462),
    }

    city = st.selectbox(
        "Select city",
        list(cities.keys()),
        key="comparison_city"
    )

    latitude, longitude = cities[city]

    try:
        with st.spinner("Loading the live forecast..."):
            weather = get_live_weather(latitude, longitude)

        daily = weather["daily"].copy()
        daily["time"] = pd.to_datetime(daily["time"])

        st.subheader("Live forecast temperature")

        daily["Forecast Max (°C)"] = daily["temperature_2m_max"]
        daily["Forecast Min (°C)"] = daily["temperature_2m_min"]

        fig = px.line(
            daily,
            x="time",
            y=["Forecast Max (°C)", "Forecast Min (°C)"],
            markers=True,
            title=f"7-Day Temperature Forecast — {city}"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            daily[
                [
                    "time",
                    "Forecast Max (°C)",
                    "Forecast Min (°C)"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Historical ML model performance")

        comparison_path = os.path.join(
            RESULTS_DIR,
            "model_comparison.csv"
        )

        if os.path.exists(comparison_path):
            comparison_df = pd.read_csv(comparison_path)

            st.write(
                "These metrics come from your existing historical "
                "model evaluation. They are not live forecast errors."
            )

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.warning(
                "Historical model comparison file was not found."
            )

        st.caption(
            "Live forecast source: Open-Meteo. Forecast maximum "
            "and minimum temperatures are not the same target as "
            "your existing next-day mean-temperature ML models. "
            "A fair accuracy comparison requires matching target "
            "definitions, dates, locations, and actual observations."
        )

    except Exception as e:
        st.error("Could not load the comparison data.")
        st.caption(f"Technical details: {e}")


elif page == "🇮🇳 India Weather Explorer":

    st.header("🇮🇳 India Weather Explorer")
    st.write(
        "Explore live weather conditions and compare temperatures "
        "across major Indian cities."
    )

    cities = {
        "Delhi": (28.6139, 77.2090),
        "Mumbai": (19.0760, 72.8777),
        "Chennai": (13.0827, 80.2708),
        "Bengaluru": (12.9716, 77.5946),
        "Kolkata": (22.5726, 88.3639),
        "Hyderabad": (17.3850, 78.4867),
        "Jaipur": (26.9124, 75.7873),
        "Lucknow": (26.8467, 80.9462),
        "Chandigarh": (30.7333, 76.7794),
        "Amritsar": (31.6340, 74.8723),
    }

    selected_cities = st.multiselect(
        "Choose cities to compare",
        options=list(cities.keys()),
        default=["Delhi", "Mumbai", "Chandigarh"],
        max_selections=10,
        key="explorer_cities",
    )

    if not selected_cities:
        st.info("Please select at least one city.")
    else:
        weather_rows = []

        with st.spinner("Fetching live weather for selected cities..."):
            for city_name in selected_cities:
                latitude, longitude = cities[city_name]

                try:
                    weather = get_live_weather(latitude, longitude)
                    current = weather["current"]
                    daily = weather["daily"]

                    weather_rows.append({
                        "City": city_name,
                        "Temperature (°C)": current["temperature_2m"],
                        "Feels Like (°C)": current["apparent_temperature"],
                        "Humidity (%)": current["relative_humidity_2m"],
                        "Wind Speed (km/h)": current["wind_speed_10m"],
                        "Forecast Max (°C)": daily.iloc[0]["temperature_2m_max"],
                        "Forecast Min (°C)": daily.iloc[0]["temperature_2m_min"],
                    })

                except Exception as e:
                    st.warning(f"Could not load weather for {city_name}: {e}")

        if weather_rows:
            explorer_df = pd.DataFrame(weather_rows)

            st.subheader("Live conditions by city")
            st.dataframe(
                explorer_df,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Current temperature comparison")
            temp_chart = px.bar(
                explorer_df.sort_values("Temperature (°C)", ascending=False),
                x="City",
                y="Temperature (°C)",
                color="Temperature (°C)",
                color_continuous_scale="RdYlBu_r",
                title="Current Temperature Across Selected Cities",
                text_auto=".1f",
            )
            st.plotly_chart(temp_chart, use_container_width=True)

            st.subheader("Today's forecast comparison")
            
            forecast_df = explorer_df.melt(
                id_vars=["City"],
                value_vars=["Forecast Max (°C)", "Forecast Min (°C)"],
                var_name="Forecast Type",
                value_name="Forecast Temperature (°C)",
            )


            forecast_chart = px.bar(
                forecast_df,
                x="City",
                y="Forecast Temperature (°C)",
                color="Forecast Type",
                barmode="group",
                title="Forecast Maximum and Minimum Temperatures",
            )
            st.plotly_chart(forecast_chart, use_container_width=True)

            st.caption(
                "Live weather data is provided by Open-Meteo. "
                "Values may change as forecasts are updated."
            )



# ============================================================
# ABOUT PROJECT - PROFESSIONAL DESIGN
# ============================================================

elif page == "🧠 About Project":

    st.markdown("""
    <style>
    .hero-box {
        padding: 42px 35px;
        border-radius: 20px;
        background: linear-gradient(120deg, #101d3a, #174b78, #287da0);
        color: white;
        margin-bottom: 25px;
    }

    .hero-box h1 {
        color: white;
        font-size: 42px;
        margin-bottom: 12px;
    }

    .hero-box p {
        color: #e1efff;
        font-size: 17px;
        line-height: 1.7;
    }

    .section-heading {
        font-size: 25px;
        font-weight: 700;
        margin-top: 30px;
        margin-bottom: 15px;
        color: inherit;
    }

    .feature-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 14px;
        padding: 20px;
        min-height: 145px;
        margin-bottom: 12px;
    }

    .feature-card h3 {
        font-size: 19px;
        margin-bottom: 8px;
    }

    .feature-card p {
        font-size: 14px;
        line-height: 1.6;
        opacity: 0.8;
    }

    .workflow-card {
        background: var(--secondary-background-color);
        border-radius: 12px;
        padding: 18px 10px;
        text-align: center;
        border: 1px solid rgba(128,128,128,0.2);
        min-height: 115px;
    }

    .workflow-card h3 {
        font-size: 17px;
    }

    .workflow-card p {
        font-size: 13px;
        opacity: 0.75;
    }

    .info-box {
        background: var(--secondary-background-color);
        padding: 22px;
        border-radius: 14px;
        border-left: 4px solid #2796c8;
        line-height: 1.8;
        margin-bottom: 15px;
    }

    .footer-box {
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        background: var(--secondary-background-color);
        margin-top: 35px;
    }
    </style>
    """, unsafe_allow_html=True)

    # HERO SECTION
    st.markdown("""
    <div class="hero-box">
        <h1>🌦️ WeatherMind AI</h1>
        <p>
            Intelligent Weather Forecasting,
            Predictive Analytics & Weather Risk Intelligence.
        </p>
        <p>
            Transforming historical weather data into meaningful
            predictions and actionable weather insights using
            Machine Learning and Deep Learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # PROJECT OVERVIEW
    st.markdown(
        '<div class="section-heading">🎯 Project Overview</div>',
        unsafe_allow_html=True
    )

    
    st.markdown(
        textwrap.dedent("""
        **WeatherMind AI** is an intelligent weather analytics
        and forecasting platform developed using Python,
        Machine Learning and Deep Learning.

        The historical prediction component uses meteorological
        data to predict the **next-day mean temperature**.

        The platform also integrates live weather forecasts,
        multi-city comparisons and configurable weather-risk
        screening.

        Its purpose is to transform weather data into meaningful
        predictions and understandable insights through
        interactive visualizations and analytical tools.
        """)
    )


    # KEY STATISTICS
    st.markdown(
        '<div class="section-heading">📊 Project at a Glance</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Historical Period", "2018–2025")
    c2.metric("Prepared Records", "2,914")
    c3.metric("ML/DL Models", "4")
    c4.metric("Input Features", "29")

    # CORE FEATURES
    st.markdown(
        '<div class="section-heading">🚀 Core Capabilities</div>',
        unsafe_allow_html=True
    )

    feature_data = [
        (
            "🌡️",
            "Temperature Prediction",
            "Predicts next-day mean temperature using trained "
            "machine learning and deep learning models."
        ),
        (
            "📈",
            "Predictive Analytics",
            "Explores model performance, prediction errors "
            "and historical weather patterns."
        ),
        (
            "🌍",
            "Live Weather Forecast",
            "Displays current conditions, hourly forecasts "
            "and seven-day weather information."
        ),
        (
            "⚠️",
            "Weather Risk Intelligence",
            "Screens upcoming forecasts for potential heat, "
            "rainfall and high-wind conditions."
        ),
        (
            "🇮🇳",
            "India Weather Explorer",
            "Compares current weather and upcoming forecasts "
            "across selected Indian cities."
        ),
        (
            "🤖",
            "AI Forecast Comparison",
            "Provides an analytical view of historical ML "
            "performance alongside live forecast information."
        ),
        (
            "🔍",
            "Feature Analysis",
            "Examines feature importance and LSTM training "
            "and validation behaviour."
        ),
        (
            "📚",
            "Historical Weather Analysis",
            "Explores historical temperature, humidity "
            "and rainfall trends."
        )
    ]

    for start in range(0, len(feature_data), 2):

        col1, col2 = st.columns(2)

        for col, item in zip(
            [col1, col2],
            feature_data[start:start + 2]
        ):

            with col:
                st.markdown(f"""
                <div class="feature-card">
                    <h3>{item[0]} {item[1]}</h3>
                    <p>{item[2]}</p>
                </div>
                """, unsafe_allow_html=True)

    # MACHINE LEARNING MODELS
    st.markdown(
        '<div class="section-heading">🧠 AI & Machine Learning Models</div>',
        unsafe_allow_html=True
    )

    model_data = [
        (
            "Linear Regression",
            "A statistical learning model used to predict "
            "temperature through relationships between input features."
        ),
        (
            "Random Forest",
            "An ensemble learning algorithm that combines "
            "multiple decision trees."
        ),
        (
            "XGBoost",
            "A gradient boosting algorithm designed to learn "
            "complex patterns from structured weather data."
        ),
        (
            "LSTM",
            "A recurrent neural network designed to learn "
            "sequential patterns in weather observations."
        )
    ]

    for start in range(0, len(model_data), 2):

        col1, col2 = st.columns(2)

        for col, item in zip(
            [col1, col2],
            model_data[start:start + 2]
        ):

            with col:
                st.markdown(f"""
                <div class="feature-card">
                    <h3>🤖 {item[0]}</h3>
                    <p>{item[1]}</p>
                </div>
                """, unsafe_allow_html=True)

    # MODEL PERFORMANCE
    st.markdown(
        '<div class="section-heading">🏆 Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    The models were evaluated on the same chronological
    test period using three standard regression metrics.
    """, unsafe_allow_html=True)

    performance = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "XGBoost",
            "LSTM"
        ],
        "MAE (°C)": [
            0.7778,
            0.7985,
            0.8873,
            1.7757
        ],
        "RMSE (°C)": [
            1.0532,
            1.0895,
            1.1996,
            2.2232
        ],
        "R²": [
            0.9729,
            0.9711,
            0.9649,
            0.8795
        ]
    })

    st.dataframe(
        performance.style.format({
            "MAE (°C)": "{:.4f}",
            "RMSE (°C)": "{:.4f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Linear Regression achieved the best overall "
        "performance among the four evaluated models "
        "on this test dataset."
    )

    st.caption(
        "Lower MAE and RMSE indicate smaller prediction errors. "
        "Higher R² indicates a better fit. These results apply "
        "to this dataset and test period."
    )

    # DATASET INFORMATION
    st.subheader("📁 Dataset & Methodology")

    st.markdown("""
    **Historical Data Source:** Open-Meteo Historical Weather API (Delhi, India).

    **Historical Period:** January 2018 to December 2025.

    **Prepared Dataset:** 2,914 records with 29 model input features.

    **Input Variables:**
    - Temperature
    - Humidity
    - Atmospheric pressure
    - Wind speed
    - Precipitation and rainfall
    - Cloud cover
    - Solar radiation
    - Calendar, lag and rolling features

    **Prediction Target:** Next-day mean temperature.

    **Data Splitting:** Chronological split consisting of 80% training data and 20% testing data.
    """)

    # SYSTEM WORKFLOW
    st.markdown(
        '<div class="section-heading">⚙️ System Workflow</div>',
        unsafe_allow_html=True
    )

    workflow = [
        ("📥", "Data Collection", "Historical weather data"),
        ("🧹", "Preprocessing", "Cleaning and feature engineering"),
        ("🧠", "AI Models", "Training and prediction"),
        ("📊", "Analysis", "Evaluation and visualization"),
        ("🌦️", "Insights", "Forecasts and risk screening")
    ]

    workflow_cols = st.columns(5)

    for col, item in zip(workflow_cols, workflow):

        with col:
            st.markdown(f"""
            <div class="workflow-card">
                <h3>{item[0]} {item[1]}</h3>
                <p>{item[2]}</p>
            </div>
            """, unsafe_allow_html=True)

    # TECHNOLOGY STACK
    st.markdown(
        '<div class="section-heading">💻 Technology Stack</div>',
        unsafe_allow_html=True
    )

    tech1, tech2, tech3 = st.columns(3)

    with tech1:
        st.markdown("""
        **Programming & Interface**

        - Python
        - Streamlit
        - HTML / CSS
        """)

    with tech2:
        st.markdown("""
        **AI & Data Science**

        - Scikit-learn
        - TensorFlow / Keras
        - XGBoost
        - Pandas
        - NumPy
        """)

    with tech3:
        st.markdown("""
        **Visualization & APIs**

        - Plotly
        - Open-Meteo API
        - Requests
        - Joblib
        """)

    # FUTURE SCOPE
    st.markdown(
        '<div class="section-heading">🔮 Future Enhancements</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    - Improved forecasting for extreme weather conditions.
    - Support for additional geographical locations.
    - Longer forecasting horizons.
    - Advanced deep learning architectures such as GRU
      and Transformers.
    - Integration of satellite and radar weather information.
    - Enhanced explainable AI and model interpretability.
    """)

    # FOOTER
    st.markdown("""
    <div class="footer-box">
        <h2>🌦️ WeatherMind AI</h2>
        <p>
            Intelligent Weather Forecasting & Risk Intelligence
        </p>
        <p>
            Academic Year 2026–27
        </p>
        <p>
            Built with Python, Machine Learning and Deep Learning
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "WeatherMind AI | Academic Year 2026–27"
)

st.sidebar.caption(
    "Machine Learning Weather Forecasting System"
)
