import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PATHS
# ============================================================

MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. FINAL MODEL METRICS
# ============================================================

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost",
        "LSTM"
    ],

    "MAE": [
        0.7778,
        0.7985,
        0.8873,
        1.3994
    ],

    "RMSE": [
        1.0532,
        1.0895,
        1.1996,
        1.7384
    ],

    "R2": [
        0.9729,
        0.9711,
        0.9649,
        0.9263
    ]
})


# ============================================================
# 3. SAVE COMPARISON TABLE
# ============================================================

comparison_path = os.path.join(
    RESULTS_DIR,
    "model_comparison.csv"
)

results.to_csv(
    comparison_path,
    index=False
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results.to_string(index=False)
)


# ============================================================
# 4. BEST MODEL
# ============================================================

best_mae = results.loc[
    results["MAE"].idxmin()
]

best_rmse = results.loc[
    results["RMSE"].idxmin()
]

best_r2 = results.loc[
    results["R2"].idxmax()
]


print("\nBest MAE:")
print(
    f"{best_mae['Model']} "
    f"({best_mae['MAE']:.4f} °C)"
)

print("\nBest RMSE:")
print(
    f"{best_rmse['Model']} "
    f"({best_rmse['RMSE']:.4f} °C)"
)

print("\nBest R²:")
print(
    f"{best_r2['Model']} "
    f"({best_r2['R2']:.4f})"
)


# ============================================================
# 5. MAE COMPARISON
# ============================================================

fig_mae = px.bar(
    results,
    x="Model",
    y="MAE",
    title="Model Comparison - Mean Absolute Error (MAE)",
    text="MAE",
    labels={
        "MAE": "MAE (°C)",
        "Model": "Model"
    }
)

fig_mae.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig_mae.update_layout(
    yaxis_title="MAE (°C)",
    xaxis_title="Model"
)

fig_mae.write_html(
    os.path.join(
        RESULTS_DIR,
        "mae_comparison.html"
    )
)


# ============================================================
# 6. RMSE COMPARISON
# ============================================================

fig_rmse = px.bar(
    results,
    x="Model",
    y="RMSE",
    title="Model Comparison - Root Mean Squared Error (RMSE)",
    text="RMSE",
    labels={
        "RMSE": "RMSE (°C)",
        "Model": "Model"
    }
)

fig_rmse.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig_rmse.update_layout(
    yaxis_title="RMSE (°C)",
    xaxis_title="Model"
)

fig_rmse.write_html(
    os.path.join(
        RESULTS_DIR,
        "rmse_comparison.html"
    )
)


# ============================================================
# 7. R2 COMPARISON
# ============================================================

fig_r2 = px.bar(
    results,
    x="Model",
    y="R2",
    title="Model Comparison - R² Score",
    text="R2",
    labels={
        "R2": "R² Score",
        "Model": "Model"
    }
)

fig_r2.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig_r2.update_layout(
    yaxis_title="R² Score",
    xaxis_title="Model",
    yaxis_range=[0, 1]
)

fig_r2.write_html(
    os.path.join(
        RESULTS_DIR,
        "r2_comparison.html"
    )
)


# ============================================================
# 8. COMBINED METRICS CHART
# ============================================================

melted = results.melt(
    id_vars="Model",
    value_vars=["MAE", "RMSE"],
    var_name="Metric",
    value_name="Value"
)

fig_metrics = px.bar(
    melted,
    x="Model",
    y="Value",
    color="Metric",
    barmode="group",
    title="MAE vs RMSE - Model Comparison",
    text="Value"
)

fig_metrics.update_traces(
    texttemplate="%{text:.4f}",
    textposition="outside"
)

fig_metrics.update_layout(
    yaxis_title="Error (°C)",
    xaxis_title="Model"
)

fig_metrics.write_html(
    os.path.join(
        RESULTS_DIR,
        "error_comparison.html"
    )
)


# ============================================================
# 9. ACTUAL VS PREDICTED
# ============================================================

prediction_files = {
    "Linear Regression":
        "baseline_predictions.csv",

    "Random Forest":
        "random_forest_predictions.csv",

    "XGBoost":
        "xgboost_predictions.csv",

    "LSTM":
        "lstm_predictions.csv"
}


for model_name, filename in prediction_files.items():

    path = os.path.join(
        MODELS_DIR,
        filename
    )

    if not os.path.exists(path):

        print(
            f"\nWarning: {filename} not found."
        )

        continue

    prediction_df = pd.read_csv(path)

    prediction_df["date"] = pd.to_datetime(
        prediction_df["date"]
    )

    # Normalize column names if necessary

    if "actual" in prediction_df.columns:

        actual_col = "actual"

    elif "actual_temperature" in prediction_df.columns:

        actual_col = "actual_temperature"

    else:

        print(
            f"Could not identify actual column "
            f"for {model_name}"
        )

        continue


    if "predicted" in prediction_df.columns:

        predicted_col = "predicted"

    elif "predicted_temperature" in prediction_df.columns:

        predicted_col = "predicted_temperature"

    else:

        print(
            f"Could not identify predicted column "
            f"for {model_name}"
        )

        continue


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=prediction_df["date"],
            y=prediction_df[actual_col],
            mode="lines",
            name="Actual Temperature"
        )
    )


    fig.add_trace(
        go.Scatter(
            x=prediction_df["date"],
            y=prediction_df[predicted_col],
            mode="lines",
            name="Predicted Temperature"
        )
    )


    fig.update_layout(
        title=f"{model_name} - Actual vs Predicted Temperature",
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode="x unified"
    )


    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )


    fig.write_html(
        os.path.join(
            RESULTS_DIR,
            f"{safe_name}_actual_vs_predicted.html"
        )
    )


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("MODEL ANALYSIS COMPLETED")
print("=" * 70)

print("\nResults folder created:")
print("results\\")

print("\nGenerated files:")

for file in sorted(os.listdir(RESULTS_DIR)):

    print(
        f"  - {file}"
    )

print("\nModel comparison completed successfully!")
