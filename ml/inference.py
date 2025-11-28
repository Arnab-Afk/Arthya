import os
import warnings
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prophet import Prophet
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

warnings.filterwarnings("ignore")


def load_dataset(csv_path: str) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	if "ds" not in df.columns or "y" not in df.columns:
		raise ValueError("Input CSV must contain 'ds' and 'y' columns")
	df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
	df = df.sort_values("ds").reset_index(drop=True)
	df = df.dropna(subset=["ds", "y"])  # ensure target/date exist
	df = df.fillna(0)
	return df


def time_series_split(df: pd.DataFrame, test_size: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
	n = len(df)
	split_idx = int(n * (1 - test_size))
	train_df = df.iloc[:split_idx].copy()
	test_df = df.iloc[split_idx:].copy()
	return train_df, test_df


def fit_prophet(train_df: pd.DataFrame, regressor_df: pd.DataFrame, feature_names: List[str]) -> Prophet:
	m = Prophet()
	safe_feature_names = [name for name in feature_names if name not in ['ds', 'y', 'cap', 'floor']]
	for name in safe_feature_names:
		m.add_regressor(name)
	train_for_prophet = pd.DataFrame({
		"ds": train_df["ds"].values,
		"y": train_df["y"].values,
		**{name: regressor_df[name].values for name in safe_feature_names},
	})
	m.fit(train_for_prophet)
	return m


def prophet_predict(m: Prophet, df: pd.DataFrame, regressor_df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
	safe_feature_names = [name for name in feature_names if name not in ['ds', 'y', 'cap', 'floor']]
	future = pd.DataFrame({
		"ds": df["ds"].values,
		**{name: regressor_df[name].values for name in safe_feature_names},
	})
	forecast = m.predict(future)
	return forecast


def create_residual_sequences(residuals: np.ndarray, regressors: np.ndarray, seq_len: int = 30) -> Tuple[np.ndarray, np.ndarray]:
	X, y = [], []
	combined_features = np.concatenate([residuals.reshape(-1, 1), regressors], axis=1)
	
	for i in range(seq_len, len(residuals)):
		X.append(combined_features[i - seq_len:i])
		y.append(residuals[i])
	return np.array(X), np.array(y)

def load_income_stats(stats_csv: str = "worker_c_hybrid.csv", income_col: str = "Income_Total") -> Tuple[float, float]:
    """
    Read stats CSV and return mean (mu) and std (sigma) of Income_Total.
    """
    if not os.path.exists(stats_csv):
        raise FileNotFoundError(f"Stats CSV not found: {stats_csv}")
    df_stats = pd.read_csv(stats_csv)
    if income_col not in df_stats.columns:
        raise ValueError(f"Column '{income_col}' not found in {stats_csv}")
    series = df_stats[income_col].dropna()
    mu = float(series.mean())
    # std() uses sample std (ddof=1) by default; change ddof=0 if you want population std.
    sigma = float(series.std())
    return mu, sigma

def income_from_standardized(pred_y: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Map standardized predictions y_pred back to income space: income = y_pred * sigma + mu
    """
    return round(pred_y * sigma + mu, -1)

def train_and_infer_on_half(
	csv_path: str = "features_c.csv",
	lstm_model_path: str = "models/lstm_residual_model.keras",
	feature_pipeline_path: str = "models/feature_pipeline.joblib",
	seq_len: int = 30,
	epochs: int = 50,
	batch_size: int = 32,
) -> Dict[str, np.ndarray]:
	print("Loading dataset...")
	df = load_dataset(csv_path)

	# Drop columns not used (aligning with retraining script to avoid leakage)
	df = df.drop(columns=['Job_Categories', 'Job_Duration_Days', 'Target_Income_Next_Day', 'Month',
						  'Monthly_Income', 'Monthly_Expenses', 'Target_Income_Next_Month', 'Income_Total'], errors='ignore')

	feature_cols = [c for c in df.columns if c not in ["ds", "y"]]
	print(f"Feature columns: {feature_cols}")

	train_df, test_df = time_series_split(df, test_size=0.8)
	print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

	# Load or build feature pipeline
	try:
		feature_pipe = joblib.load(feature_pipeline_path)
		print("Loaded existing feature pipeline.")
	except Exception as e:
		print(f"Could not load pipeline ({e}). Building new pipeline...")
		cat_cols = [c for c in feature_cols if df[c].dtype == 'object']
		num_cols = [c for c in feature_cols if c not in cat_cols]
		transformer = ColumnTransformer([
			("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
			("num", "passthrough", num_cols),
		])
		feature_pipe = Pipeline([("transform", transformer)])

	X_train_raw = train_df[feature_cols]
	X_test_raw = test_df[feature_cols]

	X_train = feature_pipe.fit_transform(X_train_raw)
	X_test = feature_pipe.transform(X_test_raw)

	X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
	X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

	transformer = feature_pipe.named_steps["transform"]
	cat_cols = transformer.transformers_[0][2] if transformer.transformers_ else []
	num_cols = transformer.transformers_[1][2] if len(transformer.transformers_) > 1 else []
	feature_names = list(cat_cols) + list(num_cols)

	reg_train_df = pd.DataFrame(X_train, columns=feature_names, index=train_df.index)
	reg_test_df = pd.DataFrame(X_test, columns=feature_names, index=test_df.index)

	print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
	print("Training Prophet model...")
	prophet_model = fit_prophet(train_df, reg_train_df, feature_names)

	forecast_train = prophet_predict(prophet_model, train_df, reg_train_df, feature_names)
	forecast_test = prophet_predict(prophet_model, test_df, reg_test_df, feature_names)

	y_train = train_df['y'].values
	y_test = test_df['y'].values
	yhat_train = forecast_train['yhat'].values
	yhat_test = forecast_test['yhat'].values

	residuals_train = y_train - yhat_train
	residuals_test = y_test - yhat_test

	print(f"Creating LSTM sequences (seq_len={seq_len})...")
	X_seq_train, y_seq_train = create_residual_sequences(residuals_train, reg_train_df.values, seq_len=seq_len)
	X_seq_test, y_seq_test = create_residual_sequences(residuals_test, reg_test_df.values, seq_len=seq_len)

	if len(X_seq_train) == 0 or len(X_seq_test) == 0:
		raise ValueError("Insufficient data for sequences. Reduce seq_len or provide more data.")

	print("Loading LSTM model...")
	try:
		lstm_model = tf.keras.models.load_model(lstm_model_path)
		print("Loaded existing LSTM model.")
		expected_shape = (X_seq_train.shape[1], X_seq_train.shape[2])
		if lstm_model.input_shape[1:] != expected_shape:
			print(f"Input shape mismatch {lstm_model.input_shape[1:]} vs {expected_shape}. Rebuilding model.")
			lstm_model = Sequential([
				LSTM(32, return_sequences=False, input_shape=expected_shape),
				Dense(16, activation='relu'),
				Dense(1),
			])
			lstm_model.compile(optimizer='adam', loss='mse')
	except Exception as e:
		print(f"Could not load LSTM model ({e}). Creating new model.")
		lstm_model = Sequential([
			LSTM(32, return_sequences=False, input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
			Dense(16, activation='relu'),
			Dense(1),
		])
		lstm_model.compile(optimizer='adam', loss='mse')

	print("Training (fine-tuning) LSTM on training sequences...")
	es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
	lstm_model.fit(
		X_seq_train,
		y_seq_train,
		validation_split=0.2,
		epochs=epochs,
		batch_size=batch_size,
		callbacks=[es],
		verbose=1,
	)

	print("Predicting residuals on test sequences...")
	pred_residuals_test = lstm_model.predict(X_seq_test, verbose=0).reshape(-1)

	aligned_yhat_test = yhat_test[seq_len:seq_len + len(pred_residuals_test)]
	aligned_y_test = y_test[seq_len:seq_len + len(pred_residuals_test)]
	final_pred_test = aligned_yhat_test + pred_residuals_test

	mu, sigma = load_income_stats(stats_csv="worker_c_hybrid.csv", income_col="Income_Total")
	prophet_income_pred = income_from_standardized(aligned_yhat_test, mu, sigma)
	combined_income_pred = income_from_standardized(final_pred_test, mu, sigma)
	
	mae = mean_absolute_error(aligned_y_test, final_pred_test)
	rmse = np.sqrt(mean_squared_error(aligned_y_test, final_pred_test))
	print(f"Test MAE: {mae:.2f}, Test RMSE: {rmse:.2f}")

	test_dates = test_df['ds'].values[seq_len:seq_len + len(pred_residuals_test)]
	
	# # Save retrained models
	# os.makedirs("models/", exist_ok=True)
	# lstm_save_path = os.path.join("models/", "lstm_residual_model.keras")
	# lstm_model.save(lstm_save_path)
	# print(f"Retrained LSTM model saved to: {lstm_save_path}")
	
	# pipeline_save_path = os.path.join("models/", "feature_pipeline.joblib")
	# joblib.dump(feature_pipe, pipeline_save_path)
	# print(f"Feature pipeline saved to: {pipeline_save_path}")

	return {
		'mae': mae,
		'rmse': rmse,
		'test_dates': test_dates,
		'test_actual': aligned_y_test,
		'prophet_predictions': aligned_yhat_test,
		'combined_predictions': final_pred_test,
		'prophet_income_pred': prophet_income_pred,   
        'combined_income_pred': combined_income_pred, 
        'income_mean': mu,                             
        'income_std': sigma,
		'seq_len': seq_len,
	}


def plot_results(results: Dict[str, np.ndarray], save_path: str = 'models/inference_half_plot.png') -> None:
	dates = results['test_dates']
	actual = results['test_actual']
	prophet_pred = results['prophet_predictions']
	combined_pred = results['combined_predictions']

	fig, axes = plt.subplots(2, 1, figsize=(14, 10))

	axes[0].plot(dates, actual, label='Actual', color='black', linewidth=2, marker='o', markersize=4)
	axes[0].plot(dates, prophet_pred, label='Prophet', color='blue', linestyle='--', linewidth=1.5)
	axes[0].plot(dates, combined_pred, label='Prophet+LSTM', color='red', linewidth=1.5)
	axes[0].set_title('Forecast Comparison (Test Half of features_c.csv)')
	axes[0].set_xlabel('Date')
	axes[0].set_ylabel('y')
	axes[0].legend()
	axes[0].grid(alpha=0.3)
	axes[0].tick_params(axis='x', rotation=45)

	prophet_residuals = actual - prophet_pred
	combined_residuals = actual - combined_pred
	axes[1].plot(dates, prophet_residuals, label='Prophet Residuals', color='blue', linewidth=1.5)
	axes[1].plot(dates, combined_residuals, label='Prophet+LSTM Residuals', color='red', linewidth=1.5)
	axes[1].axhline(0, color='black', linewidth=0.8)
	axes[1].set_title('Residual Comparison')
	axes[1].set_xlabel('Date')
	axes[1].set_ylabel('Residual')
	axes[1].legend()
	axes[1].grid(alpha=0.3)
	axes[1].tick_params(axis='x', rotation=45)

	plt.tight_layout()
	os.makedirs(os.path.dirname(save_path), exist_ok=True)
	plt.savefig(save_path, dpi=150, bbox_inches='tight')
	print(f"Plot saved to: {save_path}")
	plt.show()


if __name__ == '__main__':
	results = train_and_infer_on_half(
		csv_path='features_c.csv',
		lstm_model_path='models/lstm_residual_model.keras',
		feature_pipeline_path='models/feature_pipeline.joblib',
		seq_len=45,
		epochs=100,
		batch_size=32,
	)

	print("\nInference (Half Split) Results:")
	print(f"MAE: {results['mae']:.2f}")
	print(f"RMSE: {results['rmse']:.2f}")
	print(f"Sequence Length: {results['seq_len']}")

	plot_results(results)
	print(results["combined_income_pred"][:10])
	print(results["prophet_income_pred"][:10])