import os
import pandas as pd
import plotly.express as px


# ============================================================
# WEATHERMIND AI - FEATURE ANALYSIS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "feature_analysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("=" * 70)
print("WEATHERMIND AI - FEATURE ANALYSIS")
print("=" * 70)


# ============================================================
# 1. RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

rf_file = os.path.join(
    MODELS_DIR,
    "random_forest_feature_importance.csv"
)

if os.path.exists(rf_file):

    rf_df = pd.read_csv(rf_file)

    # Sort from highest to lowest
    rf_df = rf_df.sort_values(
        by="importance",
        ascending=False
    )

    # Save complete table
    rf_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "random_forest_feature_importance.csv"
        ),
        index=False
    )

    # Top 15 features
    rf_top = rf_df.head(15).sort_values(
        by="importance",
        ascending=True
    )

    fig_rf = px.bar(
        rf_top,
        x="importance",
        y="feature",
        orientation="h",
        title="Random Forest - Top 15 Feature Importance",
        labels={
            "importance": "Importance",
            "feature": "Feature"
        }
    )

    fig_rf.write_html(
        os.path.join(
            OUTPUT_DIR,
            "random_forest_feature_importance.html"
        )
    )

    print("\nRandom Forest feature analysis completed.")

else:
    print("\nRandom Forest feature importance file not found.")


# ============================================================
# 2. XGBOOST FEATURE IMPORTANCE
# ============================================================

xgb_file = os.path.join(
    MODELS_DIR,
    "xgboost_feature_importance.csv"
)

if os.path.exists(xgb_file):

    xgb_df = pd.read_csv(xgb_file)

    # Sort from highest to lowest
    xgb_df = xgb_df.sort_values(
        by="importance",
        ascending=False
    )

    # Save complete table
    xgb_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "xgboost_feature_importance.csv"
        ),
        index=False
    )

    # Top 15 features
    xgb_top = xgb_df.head(15).sort_values(
        by="importance",
        ascending=True
    )

    fig_xgb = px.bar(
        xgb_top,
        x="importance",
        y="feature",
        orientation="h",
        title="XGBoost - Top 15 Feature Importance",
        labels={
            "importance": "Importance",
            "feature": "Feature"
        }
    )

    fig_xgb.write_html(
        os.path.join(
            OUTPUT_DIR,
            "xgboost_feature_importance.html"
        )
    )

    print("XGBoost feature analysis completed.")

else:
    print("XGBoost feature importance file not found.")


# ============================================================
# 3. LSTM TRAINING HISTORY
# ============================================================

lstm_file = os.path.join(
    MODELS_DIR,
    "lstm_training_history.csv"
)

if os.path.exists(lstm_file):

    lstm_df = pd.read_csv(lstm_file)

    # Save copy
    lstm_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "lstm_training_history.csv"
        ),
        index=False
    )

    # Create epoch numbers because the CSV does not contain an epoch column
    lstm_df["epoch"] = range(1, len(lstm_df) + 1)

    # Plot training and validation loss
    fig_lstm = px.line(
        lstm_df,
        x="epoch",
        y=["loss", "val_loss"],
        title="LSTM Training and Validation Loss",
        labels={
            "epoch": "Epoch",
            "value": "Loss",
            "variable": "Metric"
        }
    )

    fig_lstm.write_html(
        os.path.join(
            OUTPUT_DIR,
            "lstm_training_validation_loss.html"
        )
    )

    print("LSTM training analysis completed.")

else:
    print("LSTM training history file not found.")


# ============================================================
# 4. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ANALYSIS COMPLETED")
print("=" * 70)

print(f"\nOutput folder:")
print(OUTPUT_DIR)

print("\nGenerated files:")

for file in sorted(os.listdir(OUTPUT_DIR)):
    print(" -", file)

print("\n" + "=" * 70)