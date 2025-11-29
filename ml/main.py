import requests
from flask import Flask, request, jsonify
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from prophet import Prophet
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

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


def _get_feature_pipeline(feats: pd.DataFrame) -> Tuple[sklearn.pipeline.Pipeline, List[str]]:
    """
    Build a fresh feature transformation pipeline from the provided features.
    Categorical columns are ordinal-encoded (unknowns -> -1), numeric are passed through.
    Returns the fitted pipeline along with the ordered feature names used.
    """
    feat_cols = [c for c in feats.columns if c not in ["ds", "y"]]
    cat_cols = [c for c in feat_cols if feats[c].dtype == "object"]
    num_cols = [c for c in feat_cols if c not in cat_cols]

    transformer = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols),
    ])
    feature_pipe = Pipeline([("transform", transformer)])
    feat_names = cat_cols + num_cols
    return feature_pipe, feat_names

def fill_gas_price(df: pd.DataFrame) -> pd.DataFrame:
            for idx, row in df.iterrows():
                date_str = pd.to_datetime(row['Date']).strftime('%Y-%m-%d')
                try:
                    url = "https://fuel.indianapi.in/historical_fuel_price"
                    querystring = {"location":"maharashtra", "fuel_type":"petrol", "date":date_str}
                    headers = {"X-Api-Key": os.getenv("PETROL_API", "")}
                    # TODO: Fix the URL below
                    response = requests.get(' ', headers=headers, params=querystring)

                    if response.status_code == 200:
                        price = response.json().get('price', 77.5)
                    else:
                        price = 77.5
                except Exception:
                    price = 77.5
                df.at[idx, 'Local_Gas_Price'] = price
            return df

def get_Monthly_Unemployment_Rate(date) -> float:
    """
    Dummy function to return a monthly unemployment rate based on the month.
    Replace with actual logic or data source as needed.
    """
    month = int(date.split('-')[1])
    # Example static rates for demonstration purposes
    rates = {
        1: 5.0, 2: 5.1, 3: 5.2, 4: 5.3,
        5: 5.4, 6: 5.5, 7: 5.6, 8: 5.7,
        9: 5.8, 10: 5.9, 11: 6.0, 12: 6.1
    }
    return rates.get(month, 5.0)

def _json_to_features(json_records: List[Dict]) -> pd.DataFrame:
        required_cols = [
            'Date', 'Category',
            'Platform_Count', 'Daily_Expenses', 'Daily_Income'
        ]

        df = pd.DataFrame(json_records)
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Fill Local_Gas_Price using external API if missing/zero
        df = fill_gas_price(df)
        df['Monthly_Unemployment_Rate'] = df['Date'].apply(get_Monthly_Unemployment_Rate)
        # Persist raw CSV as requested
        os.makedirs("uploads", exist_ok=True)
        raw_csv_path = os.path.join("uploads", "input.csv")
        df.to_csv(raw_csv_path, index=False)

        # Harmonize columns expected by extract_features
        df = df.sort_values('Date').reset_index(drop=True)
        df['Income_Total'] = df['Daily_Income']
        df['Expenses_Total'] = df['Daily_Expenses']
        df['Job_Categories'] = df['Category']
        
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


def _train_prophet(feats: pd.DataFrame, train_frac: float = 0.9) -> Dict[str, float]:
    global _prophet_model, _mu_sigma, _last_future_df, _last_forecast_df
    global _feature_pipe, _feature_names

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
    val_df = feats.iloc[split_idx:].copy() if split_idx < len(feats) else None

    # Instantiate a fresh, unfitted feature pipeline from current features
    feature_pipe, feat_names = _get_feature_pipeline(feats)

    # Fit the pipeline on training data and transform
    feature_pipe.fit(train_df[feat_names])
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
    _feature_pipe = feature_pipe
    _feature_names = feat_names

    # Calculate accuracy metrics on training data
    train_pred = m.predict(train_prophet_df)
    train_y_actual = train_df['y'].values
    train_y_pred = train_pred['yhat'].values
    
    train_mae = float(np.mean(np.abs(train_y_actual - train_y_pred)))
    train_rmse = float(np.sqrt(np.mean((train_y_actual - train_y_pred) ** 2)))
    train_r2 = float(1 - np.sum((train_y_actual - train_y_pred) ** 2) / np.sum((train_y_actual - np.mean(train_y_actual)) ** 2))
    
    metrics = {
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
    }

    # Calculate validation metrics if validation set exists
    if val_df is not None and len(val_df) > 0:
        reg_val_df = pd.DataFrame(
            feature_pipe.transform(val_df[feat_names]),
            columns=feat_names,
            index=val_df.index,
        )
        val_prophet_df = pd.DataFrame(
            {
                'ds': val_df['ds'].values,
                **{name: reg_val_df[name].values for name in safe_regressors},
            }
        )
        val_pred = m.predict(val_prophet_df)
        val_y_actual = val_df['y'].values
        val_y_pred = val_pred['yhat'].values
        
        val_mae = float(np.mean(np.abs(val_y_actual - val_y_pred)))
        val_rmse = float(np.sqrt(np.mean((val_y_actual - val_y_pred) ** 2)))
        val_r2 = float(1 - np.sum((val_y_actual - val_y_pred) ** 2) / np.sum((val_y_actual - np.mean(val_y_actual)) ** 2))
        
        metrics.update({
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
        })

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
    
    return metrics


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
        metrics = _train_prophet(feats, train_frac=0.9)

        # Prepare outputs for saving
        mu, sigma = _mu_sigma  # type: ignore
        forecast_df = _last_forecast_df  # type: ignore
        dates = forecast_df['ds'].dt.strftime('%Y-%m-%d').tolist()
        yhat = forecast_df['yhat'].astype(float).tolist()
        incomes = [float(income_from_standardized(float(v), mu, sigma)) for v in yhat]
        trend = forecast_df['trend'].astype(float).tolist() if 'trend' in forecast_df.columns else [0.0] * len(forecast_df)
        weekly = forecast_df['weekly'].astype(float).tolist() if 'weekly' in forecast_df.columns else [0.0] * len(forecast_df)
        yearly = forecast_df['yearly'].astype(float).tolist() if 'yearly' in forecast_df.columns else [0.0] * len(forecast_df)

        # Save to CSV
        out_df = pd.DataFrame({
            'date': dates,
            'predicted_income': incomes,
            'trend': trend,
            'weekly': weekly,
            'yearly': yearly,
        })
        os.makedirs("uploads", exist_ok=True)
        out_df.to_csv(os.path.join("uploads", "predictions.csv"), index=False)

        return jsonify({
            "status": "trained",
            "rows": len(feats),
            "train_rows": int(len(feats) * 0.9),
            "message": "Prophet model trained and 7-day forecast prepared.",
            "csv_path": "uploads/predictions.csv",
            "accuracy": {
                "train": {
                    "mae": round(metrics.get('train_mae', 0), 4),
                    "rmse": round(metrics.get('train_rmse', 0), 4),
                    "r2": round(metrics.get('train_r2', 0), 4)
                },
                "validation": {
                    "mae": round(metrics.get('val_mae', 0), 4),
                    "rmse": round(metrics.get('val_rmse', 0), 4),
                    "r2": round(metrics.get('val_r2', 0), 4)
                } if 'val_mae' in metrics else None
            }
        })
    except Exception as e:
        return jsonify({"Error": str(e)}), 500


@app.get("/predict_7_days")
def predict_7_days():
    try:
        csv_path = os.path.join("uploads", "predictions.csv")
        if not os.path.exists(csv_path):
            return jsonify({"error": "No predictions available. Train first."}), 404
        df = pd.read_csv(csv_path)
        dates = df['date'].astype(str).tolist()
        incomes = df['predicted_income'].astype(float).tolist()
        return jsonify({
            "dates": dates,
            "income": incomes,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/growth_trend")
def growth_trend():
    try:
        csv_path = os.path.join("uploads", "predictions.csv")
        if not os.path.exists(csv_path):
            return jsonify({"error": "No predictions available. Train first."}), 404
        df = pd.read_csv(csv_path)
        dates = df['date'].astype(str).tolist()
        trend = df['trend'].astype(float).tolist() if 'trend' in df.columns else [0.0] * len(df)
        return jsonify({
            "dates": dates,
            "trend": trend,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/seasonality")
def seasonality():
    try:
        csv_path = os.path.join("uploads", "predictions.csv")
        if not os.path.exists(csv_path):
            return jsonify({"error": "No predictions available. Train first."}), 404
        df = pd.read_csv(csv_path)
        dates = df['date'].astype(str).tolist()
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