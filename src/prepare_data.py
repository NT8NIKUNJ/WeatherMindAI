import pandas as pd
from pathlib import Path


# ============================================================
# WeatherMind AI - Data Preparation & Feature Engineering
# ============================================================

INPUT_FILE = Path("data/weather_delhi_2018_2025.csv")
OUTPUT_FILE = Path("data/weather_model_ready.csv")


def prepare_weather_data():

    print("🌦️ WeatherMind AI - Data Preparation")
    print("-" * 55)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("📂 Loading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original rows    : {len(df)}")
    print(f"Original columns : {len(df.columns)}")

    # --------------------------------------------------------
    # 2. Convert date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    # Check duplicate dates
    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in dataset!")

    # Check missing calendar dates
    expected_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D"
    )

    missing_dates = expected_dates.difference(df["date"])

    if len(missing_dates) > 0:
        raise ValueError(
            f"Missing dates found: {len(missing_dates)}"
        )

    print("Date continuity check passed!")

    # --------------------------------------------------------
    # 3. Calendar features
    # --------------------------------------------------------

    print("📅 Creating calendar features...")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek

    # --------------------------------------------------------
    # 4. Lag features
    # --------------------------------------------------------

    print("🔄 Creating lag features...")

    df["temperature_lag_1"] = df["temperature_mean"].shift(1)
    df["temperature_lag_3"] = df["temperature_mean"].shift(3)
    df["temperature_lag_7"] = df["temperature_mean"].shift(7)

    df["humidity_lag_1"] = df["humidity_mean"].shift(1)
    df["pressure_lag_1"] = df["pressure"].shift(1)
    df["rainfall_lag_1"] = df["rainfall"].shift(1)
    df["wind_speed_lag_1"] = df["wind_speed_mean"].shift(1)

    # --------------------------------------------------------
    # 5. Rolling features
    # --------------------------------------------------------

    print("📊 Creating rolling features...")

    df["temperature_rolling_3"] = (
        df["temperature_mean"]
        .shift(1)
        .rolling(window=3)
        .mean()
    )

    df["temperature_rolling_7"] = (
        df["temperature_mean"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    df["humidity_rolling_7"] = (
        df["humidity_mean"]
        .shift(1)
        .rolling(window=7)
        .mean()
    )

    # 6. Create prediction target

    print("🎯 Creating target variable...")

    # Date of the temperature being predicted
    df["target_date"] = df["date"].shift(-1)

    # Tomorrow's mean temperature
    df["target_temperature"] = df["temperature_mean"].shift(-1) 

    # --------------------------------------------------------
    # 7. Remove rows created by lag/target operations
    # --------------------------------------------------------

    before = len(df)

    df.dropna(inplace=True)

    after = len(df)

    print(f"Rows removed     : {before - after}")
    print(f"Final rows       : {after}")

    # --------------------------------------------------------
    # 8. Save model-ready dataset
    # --------------------------------------------------------

    df.to_csv(OUTPUT_FILE, index=False)

    print()
    print("✅ Feature engineering completed!")
    print(f"📁 Output file: {OUTPUT_FILE}")
    print(f"📊 Final shape: {df.shape}")

    print()
    print("📋 Feature columns:")
    print(df.columns.tolist())

    print()
    print("🎯 Target statistics:")
    print(df["target_temperature"].describe())

    print()
    print("🔍 Missing values:")
    print(df.isnull().sum())

    print()
    print("🎉 Dataset is ready for model training!")


if __name__ == "__main__":
    prepare_weather_data()