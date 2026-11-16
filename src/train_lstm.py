import os
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/weather_model_ready.csv"
MODEL_PATH = "models/lstm_model.keras"
PREDICTIONS_PATH = "models/lstm_predictions.csv"
FEATURE_SCALER_PATH = "models/lstm_feature_scaler.pkl"
TARGET_SCALER_PATH = "models/lstm_target_scaler.pkl"
HISTORY_PATH = "models/lstm_training_history.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading prepared weather dataset...")

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df["target_date"] = pd.to_datetime(df["target_date"])

df = (
    df.sort_values("date")
    .reset_index(drop=True)
)

print(f"Dataset shape: {df.shape}")

print(
    f"Date range: "
    f"{df['date'].min().date()} -> "
    f"{df['date'].max().date()}"
)


# ============================================================
# 3. FEATURES
# ============================================================

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

target_column = "target_temperature"


X_raw = df[feature_columns].values
y_raw = df[target_column].values.reshape(-1, 1)


# ============================================================
# 4. CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(len(df) * 0.80)

print("\nChronological split:")

print(f"Training rows: {split_index}")
print(f"Testing rows : {len(df) - split_index}")

print(
    f"Training period: "
    f"{df['target_date'].iloc[0].date()} -> "
    f"{df['target_date'].iloc[split_index - 1].date()}"
)

print(
    f"Testing period : "
    f"{df['target_date'].iloc[split_index].date()} -> "
    f"{df['target_date'].iloc[-1].date()}"
)


# ============================================================
# 5. SCALE FEATURES
# ============================================================

print("\nScaling features and target...")

feature_scaler = MinMaxScaler()

target_scaler = MinMaxScaler()


# IMPORTANT:
# Fit scalers ONLY on training data

X_train_raw = X_raw[:split_index]

y_train_raw = y_raw[:split_index]


feature_scaler.fit(X_train_raw)

target_scaler.fit(y_train_raw)


# Transform complete dataset using
# scalers learned from training data

X_scaled = feature_scaler.transform(X_raw)

y_scaled = target_scaler.transform(y_raw)


# Save scalers

os.makedirs("models", exist_ok=True)

joblib.dump(
    feature_scaler,
    FEATURE_SCALER_PATH
)

joblib.dump(
    target_scaler,
    TARGET_SCALER_PATH
)


# ============================================================
# 6. CREATE TIME SERIES SEQUENCES
# ============================================================

SEQUENCE_LENGTH = 7


def create_sequences(
    X_data,
    y_data,
    dates,
    sequence_length
):

    X_sequences = []
    y_sequences = []
    sequence_dates = []

    for i in range(
        sequence_length - 1,
        len(X_data)
    ):

        # Previous 7 days including current day
        X_sequences.append(
            X_data[
                i - sequence_length + 1:i + 1
            ]
        )

        # Target = next-day temperature
        y_sequences.append(
            y_data[i]
        )

        sequence_dates.append(
            dates.iloc[i]
        )

    return (
        np.array(X_sequences),
        np.array(y_sequences),
        pd.Series(sequence_dates)
    )


dates = df['target_date'].reset_index(drop=True)


X_sequences, y_sequences, sequence_dates = create_sequences(
    X_scaled,
    y_scaled,
    dates,
    SEQUENCE_LENGTH
)


# ============================================================
# 7. SPLIT SEQUENCES CHRONOLOGICALLY
# ============================================================

# Sequence target corresponding to original row index
sequence_target_indices = np.arange(
    SEQUENCE_LENGTH - 1,
    len(df)
)


train_mask = (
    sequence_target_indices < split_index 
)

test_mask = (
    sequence_target_indices >= split_index
)


X_train = X_sequences[train_mask]

y_train = y_sequences[train_mask]

X_test = X_sequences[test_mask]

y_test = y_sequences[test_mask]

test_dates = sequence_dates[test_mask].reset_index(drop=True)


print("\nSequence information:")

print(f"Sequence length: {SEQUENCE_LENGTH} days")

print(f"X_train shape: {X_train.shape}")

print(f"y_train shape: {y_train.shape}")

print(f"X_test shape : {X_test.shape}")

print(f"y_test shape : {y_test.shape}")


# ============================================================
# 8. BUILD LSTM
# ============================================================

print("\nBuilding LSTM model...")


model = Sequential([

    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(
        32,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(1)
])


model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


model.summary()


# ============================================================
# 9. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


# ============================================================
# 10. TRAIN
# ============================================================

print("\nTraining LSTM model...")

history = model.fit(

    X_train,
    y_train,

    epochs=50,

    batch_size=32,

    validation_split=0.15,

    callbacks=[
        early_stopping
    ],

    shuffle=False,

    verbose=1
)


# ============================================================
# 11. PREDICTIONS
# ============================================================

print("\nGenerating predictions...")


predictions_scaled = model.predict(
    X_test,
    verbose=0
)


# Convert predictions back to Celsius

predictions = target_scaler.inverse_transform(
    predictions_scaled
).flatten()


# Convert actual values back to Celsius

actual = target_scaler.inverse_transform(
    y_test
).flatten()

# ============================================================
# PERSISTENCE BASELINE
# ============================================================

test_target_indices = sequence_target_indices[test_mask]

baseline_predictions = df.loc[
    test_target_indices,
    "temperature_mean"
].to_numpy()

baseline_mae = mean_absolute_error(
    actual,
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        actual,
        baseline_predictions
    )
)

baseline_r2 = r2_score(
    actual,
    baseline_predictions
)

print("\nPERSISTENCE BASELINE PERFORMANCE")
print("=" * 60)

print(f"MAE  : {baseline_mae:.4f} °C")
print(f"RMSE : {baseline_rmse:.4f} °C")
print(f"R²   : {baseline_r2:.4f}")

# ============================================================
# 12. EVALUATION
# ============================================================

mae = mean_absolute_error(
    actual,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)

r2 = r2_score(
    actual,
    predictions
)


print("\n" + "=" * 60)

print("CORRECTED LSTM MODEL PERFORMANCE")

print("=" * 60)

print(f"MAE  : {mae:.4f} °C")

print(f"RMSE : {rmse:.4f} °C")

print(f"R²   : {r2:.4f}")

print("=" * 60)


# ============================================================
# 13. SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "date": test_dates,

    "actual_temperature": actual,

    "predicted_temperature": predictions

})


prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# 14. SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)


# ============================================================
# 15. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# ============================================================
# 16. SAMPLE PREDICTIONS
# ============================================================

print("\nSample predictions:")

print(
    prediction_df
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 17. FILES
# ============================================================

print("\nFiles saved successfully:")

print(
    f"Model           : {MODEL_PATH}"
)

print(
    f"Predictions     : {PREDICTIONS_PATH}"
)

print(
    f"Feature scaler  : {FEATURE_SCALER_PATH}"
)

print(
    f"Target scaler   : {TARGET_SCALER_PATH}"
)

print(
    f"Training history: {HISTORY_PATH}"
)

print("\nCorrected LSTM training completed successfully!")