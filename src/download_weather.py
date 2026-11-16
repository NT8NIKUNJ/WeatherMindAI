import requests
import pandas as pd
from pathlib import Path


# ============================================================
# WeatherMind AI - Historical Weather Data Collection
# ============================================================

LATITUDE = 28.6139
LONGITUDE = 77.2090

START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

API_URL = "https://archive-api.open-meteo.com/v1/archive"


def download_weather_data():

    print("🌦️ WeatherMind AI - Data Collection")
    print("-" * 50)
    print(f"Location : Delhi, India")
    print(f"Period   : {START_DATE} to {END_DATE}")
    print()

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,

        "daily": ",".join([
            "temperature_2m_mean",
            "temperature_2m_max",
            "temperature_2m_min",
            "relative_humidity_2m_mean",
            "relative_humidity_2m_max",
            "relative_humidity_2m_min",
            "dew_point_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "pressure_msl_mean",
            "wind_speed_10m_mean",
            "wind_speed_10m_max",
            "cloud_cover_mean",
            "shortwave_radiation_sum"
        ]),

        "timezone": "Asia/Kolkata",

        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm"
    }

    print("📡 Requesting historical weather data...")

    response = requests.get(
        API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "daily" not in data:
        raise RuntimeError("No daily weather data returned by API.")

    daily_data = data["daily"]

    df = pd.DataFrame(daily_data)

    # Rename columns to cleaner project names
    rename_columns = {
        "time": "date",
        "temperature_2m_mean": "temperature_mean",
        "temperature_2m_max": "temperature_max",
        "temperature_2m_min": "temperature_min",
        "relative_humidity_2m_mean": "humidity_mean",
        "relative_humidity_2m_max": "humidity_max",
        "relative_humidity_2m_min": "humidity_min",
        "dew_point_2m_mean": "dewpoint_mean",
        "precipitation_sum": "precipitation",
        "rain_sum": "rainfall",
        "pressure_msl_mean": "pressure",
        "wind_speed_10m_mean": "wind_speed_mean",
        "wind_speed_10m_max": "wind_speed_max",
        "cloud_cover_mean": "cloud_cover",
        "shortwave_radiation_sum": "solar_radiation"
    }

    df.rename(columns=rename_columns, inplace=True)

    # Convert date column
    df["date"] = pd.to_datetime(df["date"])

    # Sort chronologically
    df.sort_values("date", inplace=True)

    # Remove duplicate dates
    df.drop_duplicates(subset=["date"], inplace=True)

    # Reset index
    df.reset_index(drop=True, inplace=True)

    # Create data directory if it doesn't exist
    data_directory = Path("data")
    data_directory.mkdir(exist_ok=True)

    output_file = data_directory / "weather_delhi_2018_2025.csv"

    df.to_csv(output_file, index=False)

    print()
    print("✅ Dataset downloaded successfully!")
    print(f"📁 File: {output_file}")
    print(f"📊 Rows: {len(df)}")
    print(f"📊 Columns: {len(df.columns)}")
    print()

    print("Dataset columns:")
    print(df.columns.tolist())

    print()
    print("First 5 records:")
    print(df.head())

    print()
    print("Missing values:")
    print(df.isnull().sum())

    print()
    print("🎉 Data collection completed!")


if __name__ == "__main__":
    download_weather_data()