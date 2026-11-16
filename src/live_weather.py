
import requests
import pandas as pd
import streamlit as st


FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


@st.cache_data(ttl=900, show_spinner=False)
def get_live_weather(latitude, longitude):
    """Fetch and cache a location's weather forecast."""

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "is_day",
            "precipitation",
            "rain",
            "showers",
            "weather_code",
            "wind_speed_10m",
        ],
        "hourly": [
            "temperature_2m",
            "precipitation_probability",
            "precipitation",
            "relative_humidity_2m",
            "wind_speed_10m",
        ],
        "daily": [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "precipitation_sum",
            "wind_speed_10m_max",
        ],
        "timezone": "auto",
        "forecast_days": 7,
    }

    response = requests.get(
        FORECAST_URL,
        params=params,
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()

    return {
        "current": data["current"],
        "hourly": pd.DataFrame(data["hourly"]),
        "daily": pd.DataFrame(data["daily"]),
        "timezone": data.get("timezone", "Local time"),
        "fetched_at": pd.Timestamp.now(tz="UTC"),
    }
