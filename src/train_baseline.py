import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# WeatherMind AI - Baseline Model
# Linear Regression for Next-Day Temperature Forecasting
# ============================================================

INPUT_FILE = Path("data/weather_model_ready.csv")


def train_baseline():

    print("🌦️ WeatherMind AI - Baseline Model")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load prepared dataset
    # --------------------------------------------------------

    print("📂 Loading model-ready dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df["target_date"] = pd.to_datetime(df["target_date"])

    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # 2. Define target
    # --------------------------------------------------------

    target = "target_temperature"

    # --------------------------------------------------------
    # 3. Select features
    # --------------------------------------------------------

    feature_columns = [
        "temperature_mean",
        "temperature_max",
        "temperature_min",

        "humidity_mean",
        "humidity_max",
        "humidity_min",

        "dewpoint_mean",

        "precipitation",
        "rainfall",

        "pressure",

        "wind_speed_mean",
        "wind_speed_max",

        "cloud_cover",
        "solar_radiation",

        "year",
        "month",
        "day",
        "day_of_year",
        "day_of_week",

        "temperature_lag_1",
        "temperature_lag_3",
        "temperature_lag_7",

        "humidity_lag_1",
        "pressure_lag_1",
        "rainfall_lag_1",
        "wind_speed_lag_1",

        "temperature_rolling_3",
        "temperature_rolling_7",
        "humidity_rolling_7"
    ]

    X = df[feature_columns]
    y = df[target]

    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples : {len(X)}")

    # --------------------------------------------------------
    # 4. Time-based train/test split
    # --------------------------------------------------------

    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    train_dates = df["target_date"].iloc[:split_index]
    test_dates = df["target_date"].iloc[split_index:]

    print()
    print("📊 Time-based split")
    print("-" * 40)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print()
    print(f"Training period: {train_dates.iloc[0].date()} → "
          f"{train_dates.iloc[-1].date()}")

    print(f"Testing period : {test_dates.iloc[0].date()} → "
          f"{test_dates.iloc[-1].date()}")

    # --------------------------------------------------------
    # 5. Train Linear Regression
    # --------------------------------------------------------

    print()
    print("🤖 Training Linear Regression...")

    model = LinearRegression()

    model.fit(X_train, y_train)

    print("✅ Model training completed!")

    # --------------------------------------------------------
    # 6. Predictions
    # --------------------------------------------------------

    print()
    print("🔮 Generating predictions...")

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Persistence Baseline
    # --------------------------------------------------------

    baseline_predictions = X_test["temperature_mean"].values

    baseline_mae = mean_absolute_error(
        y_test, baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(y_test, baseline_predictions)
    )

    baseline_r2 = r2_score(
        y_test, baseline_predictions
    )

    print()
    print("📊 PERSISTENCE BASELINE PERFORMANCE")
    print("=" * 60)

    print(f"MAE  : {baseline_mae:.4f} °C")
    print(f"RMSE : {baseline_rmse:.4f} °C")
    print(f"R²   : {baseline_r2:.4f}")

    # --------------------------------------------------------
    # 7. Evaluation
    # --------------------------------------------------------

    mae = mean_absolute_error(y_test, predictions)

    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    r2 = r2_score(y_test, predictions)

    print()
    print("📈 MODEL PERFORMANCE")
    print("=" * 60)

    print(f"MAE  : {mae:.4f} °C")
    print(f"RMSE : {rmse:.4f} °C")
    print(f"R²   : {r2:.4f}")

    # --------------------------------------------------------
    # 8. Sample predictions
    # --------------------------------------------------------

    results = pd.DataFrame({
        "date": test_dates.values,
        "actual_temperature": y_test.values,
        "predicted_temperature": predictions
    })

    print()
    print("🔍 Sample predictions:")
    print("-" * 60)

    print(results.head(10).to_string(index=False))

    # --------------------------------------------------------
    # 9. Save predictions
    # --------------------------------------------------------

    output_directory = Path("models")
    output_directory.mkdir(exist_ok=True)

    results_file = output_directory / "baseline_predictions.csv"

    results.to_csv(results_file, index=False)

    print()
    print(f"💾 Predictions saved to: {results_file}")

    # --------------------------------------------------------
    # 10. Save model
    # --------------------------------------------------------

    import joblib

    model_file = output_directory / "linear_regression.pkl"

    joblib.dump(model, model_file)

    print(f"💾 Model saved to: {model_file}")

    print()
    print("🎉 Baseline model completed!")


if __name__ == "__main__":
    train_baseline()