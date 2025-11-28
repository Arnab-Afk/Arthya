from flask import Flask, request, jsonify
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from prophet import Prophet

from featureExtraction import extract_features
from inference import income_from_standardized


app = Flask(__name__)


# Global state to reuse between requests
FEATURE_PIPELINE_PATH = "models/feature_pipeline.joblib"
_feature_pipe = None
_feature_names: List[str] | None = None
_prophet_model: Prophet | None = None
_mu_sigma: Tuple[float, float] | None = None
_last_future_df: pd.DataFrame | None = None
_last_forecast_df: pd.DataFrame | None = None


def _get_feature_pipeline() -> Tuple[object, List[str]]:
    global _feature_pipe, _feature_names
    if _feature_pipe is None:
        if not os.path.exists(FEATURE_PIPELINE_PATH):
            raise FileNotFoundError(f"Feature pipeline not found at {FEATURE_PIPELINE_PATH}")
        _feature_pipe = joblib.load(FEATURE_PIPELINE_PATH)
        transformer = _feature_pipe.named_steps.get("transform")
        if transformer is None:
            raise ValueError("Loaded feature pipeline missing 'transform' step")
        cat_cols = transformer.transformers_[0][2] if transformer.transformers_ else []
        num_cols = transformer.transformers_[1][2] if len(transformer.transformers_) > 1 else []
        _feature_names = list(cat_cols) + list(num_cols)
    return _feature_pipe, _feature_names or []


def _json_to_features(json_records: List[Dict]) -> pd.DataFrame:
    required_cols = [
        'Date', 'Job_Type', 'Education_Level', 'Public_Holiday_Flag',
        'Local_Gas_Price', 'Monthly_Unemployment_Rate', 'Hours_Worked',
        'Platform_Count', 'Jobs_Completed', 'Daily_Expenses', 'Daily_Income'
    ]

    df = pd.DataFrame(json_records)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Persist raw CSV as requested
    os.makedirs("uploads", exist_ok=True)
    raw_csv_path = os.path.join("uploads", "input.csv")
    df.to_csv(raw_csv_path, index=False)

    # Harmonize columns expected by extract_features
    df = df.sort_values('Date').reset_index(drop=True)
    df['Income_Total'] = df['Daily_Income']
    df['Expenses_Total'] = df['Daily_Expenses']
    # Approximate monthly aggregates with 30-day rolling sums
    df['Monthly_Income'] = df['Daily_Income'].rolling(window=30, min_periods=1).sum()
    df['Monthly_Expenses'] = df['Daily_Expenses'].rolling(window=30, min_periods=1).sum()

    # Apply provided feature extraction
    feats = extract_features(df)
    feats = feats.rename(columns={'Date': 'ds'})
    feats['ds'] = pd.to_datetime(feats['ds'], errors='coerce')
    feats = feats.dropna(subset=['ds']).reset_index(drop=True)

    # Save features for traceability
    feats.to_csv(os.path.join("uploads", "features.csv"), index=False)
    return feats


def _train_prophet(feats: pd.DataFrame, train_frac: float = 0.9) -> None:
    global _prophet_model, _mu_sigma, _last_future_df, _last_forecast_df

    feats = feats.sort_values('ds').reset_index(drop=True)

    if 'Income_Total' not in feats.columns:
        raise ValueError("'Income_Total' is required in features for standardization stats")

    mu = float(feats['Income_Total'].mean())
    # population std to match extract_features default
    sigma = float(feats['Income_Total'].std(ddof=0)) or 1.0
    _mu_sigma = (mu, sigma)

    if 'y' not in feats.columns:
        feats['y'] = (feats['Income_Total'] - mu) / sigma

    split_idx = max(1, int(len(feats) * train_frac))
    train_df = feats.iloc[:split_idx].copy()

    feature_pipe, feat_names = _get_feature_pipeline()
    # Ensure all expected feature columns exist; backfill with zeros if missing
    for col in feat_names:
        if col not in feats.columns:
            feats[col] = 0
            train_df[col] = 0

    # Transform with the prefit pipeline (do not refit)
    reg_train_df = pd.DataFrame(
        feature_pipe.transform(train_df[feat_names]),
        columns=feat_names,
        index=train_df.index,
    )

    # Fit Prophet with regressors
    safe_regressors = [c for c in feat_names if c not in ['ds', 'y', 'cap', 'floor']]
    m = Prophet()
    for name in safe_regressors:
        m.add_regressor(name)

    train_prophet_df = pd.DataFrame(
        {
            'ds': train_df['ds'].values,
            'y': train_df['y'].values,
            **{name: reg_train_df[name].values for name in safe_regressors},
        }
    )
    m.fit(train_prophet_df)
    _prophet_model = m

    # Prepare 7-day future with last known regressors repeated
    last_row = feats.iloc[[-1]][feat_names]
    last_reg = pd.DataFrame(
        feature_pipe.transform(last_row), columns=feat_names, index=[0]
    )
    future_dates = pd.date_range(start=feats['ds'].max() + pd.Timedelta(days=1), periods=7, freq='D')
    reg_future_df = pd.DataFrame(
        np.repeat(last_reg.values, len(future_dates), axis=0),
        columns=feat_names,
    )
    safe_future = pd.DataFrame({'ds': future_dates})
    for name in safe_regressors:
        safe_future[name] = reg_future_df[name].values

    forecast = m.predict(safe_future)
    _last_future_df = safe_future
    _last_forecast_df = forecast


def _ensure_trained():
    if _prophet_model is None or _last_forecast_df is None or _mu_sigma is None:
        raise RuntimeError("Model not trained yet. POST data to /train first.")


@app.post("/train")
def train_endpoint():
    try:
        payload = request.get_json(force=True, silent=False)
        # Accept either a list of records or an object with 'data'
        records = payload if isinstance(payload, list) else payload.get('data', [])
        if not isinstance(records, list) or len(records) == 0:
            return jsonify({"error": "Provide a non-empty JSON array or {'data': [...]}"}), 400

        feats = _json_to_features(records)
        _train_prophet(feats, train_frac=0.9)
        return jsonify({
            "status": "trained",
            "rows": len(feats),
            "train_rows": int(len(feats) * 0.9),
            "message": "Prophet model trained and 7-day forecast prepared."
        })
    except Exception as e:
        return jsonify({"Error": str(e)}), 500


@app.get("/predict_7_days")
def predict_7_days():
    try:
        _ensure_trained()
        mu, sigma = _mu_sigma  # type: ignore

        dates = _last_forecast_df['ds'].dt.strftime('%Y-%m-%d').tolist()  # type: ignore
        yhat = _last_forecast_df['yhat'].astype(float).tolist()  # type: ignore

        # income_from_standardized expects scalars; apply element-wise
        incomes = [float(income_from_standardized(float(v), mu, sigma)) for v in yhat]
        return jsonify({
            "dates": dates,
            "income": incomes,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/growth_trend")
def growth_trend():
    try:
        _ensure_trained()
        dates = _last_forecast_df['ds'].dt.strftime('%Y-%m-%d').tolist()  # type: ignore
        trend = _last_forecast_df['trend'].astype(float).tolist()  # type: ignore
        return jsonify({
            "dates": dates,
            "trend": trend,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/seasonality")
def seasonality():
    try:
        _ensure_trained()
        df = _last_forecast_df  # type: ignore
        dates = df['ds'].dt.strftime('%Y-%m-%d').tolist()
        weekly = df['weekly'].astype(float).tolist() if 'weekly' in df.columns else [0.0] * len(df)
        yearly = df['yearly'].astype(float).tolist() if 'yearly' in df.columns else [0.0] * len(df)
        return jsonify({
            "dates": dates,
            "weekly": weekly,
            "yearly": yearly,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Default to port 5000; configurable via PORT env var
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
