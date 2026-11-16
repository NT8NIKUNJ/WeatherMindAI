import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# WeatherMind AI - XGBoost Model
# Next-Day Temperature Forecasting
# ===========================================================

INPUT_FILE = Path("data/weather_model_ready.csv")


def train_xgboost():

    print("🌦️ WeatherMind AI - XGBoost")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("📂 Loading model-ready dataset...")

    df = pd.read_csv(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    print(f"Dataset shape: {df.shape}")

    # --------------------------------------------------------
    # 2. Features and target
    # --------------------------------------------------------

    target = "target_temperature"

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

    # --------------------------------------------------------
    # 3. Time-based split
    # --------------------------------------------------------

    split_index = int(len(df) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    test_dates = df["target_date"].iloc[split_index:]

    print()
    print("📊 Time-based split")
    print("-" * 40)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    # --------------------------------------------------------
    # 4. Create XGBoost model
    # --------------------------------------------------------

    print()
    print("🚀 Creating XGBoost model...")

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # 5. Train model
    # --------------------------------------------------------

    print("🤖 Training XGBoost...")
    print("This may take a little time...")

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

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print()
    print("📈 XGBOOST PERFORMANCE")
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

    print(
        results.head(10).to_string(index=False)
    )

    # --------------------------------------------------------
    # 9. Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    print()
    print("🔎 TOP 10 IMPORTANT FEATURES")
    print("-" * 60)

    print(
        importance.head(10).to_string(index=False)
    )

    # --------------------------------------------------------
    # 10. Save model
    # --------------------------------------------------------

    output_directory = Path("models")
    output_directory.mkdir(exist_ok=True)

    model_file = (
        output_directory /
        "xgboost_model.pkl"
    )

    joblib.dump(
        model,
        model_file
    )

    print()
    print(
        f"💾 Model saved to: {model_file}"
    )

    # --------------------------------------------------------
    # 11. Save predictions
    # --------------------------------------------------------

    predictions_file = (
        output_directory /
        "xgboost_predictions.csv"
    )

    results.to_csv(
        predictions_file,
        index=False
    )

    print(
        f"💾 Predictions saved to: "
        f"{predictions_file}"
    )

    # --------------------------------------------------------
    # 12. Save feature importance
    # --------------------------------------------------------

    importance_file = (
        output_directory /
        "xgboost_feature_importance.csv"
    )

    importance.to_csv(
        importance_file,
        index=False
    )

    print(
        f"💾 Feature importance saved to: "
        f"{importance_file}"
    )

    print()
    print("🎉 XGBoost completed!")


if __name__ == "__main__":
    train_xgboost()