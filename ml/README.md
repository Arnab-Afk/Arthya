# ML Module - Income Prediction System

## Overview

This machine learning module provides income prediction capabilities for gig economy workers using a hybrid Prophet + LSTM architecture. The system is designed to forecast future income based on historical transaction data, economic indicators, and worker behavior patterns.

## Architecture

The prediction system combines two complementary approaches:

- **Facebook Prophet**: Handles trend and seasonality decomposition
- **LSTM Neural Network**: Captures residual patterns and complex temporal dependencies

## Features

### Core Capabilities

- 7-day income forecasting
- Growth trend analysis
- Seasonality detection (weekly and yearly patterns)
- Support for multiple worker types (drivers, freelancers, hybrid workers)
- Real-time predictions via REST API

### Data Processing

- Feature extraction and engineering
- Automated data standardization
- External API integration (fuel prices, unemployment rates)
- Synthetic data generation for testing

## Project Structure

```
ml/
├── main.py                      # Flask API server
├── dataGenerator.py             # Synthetic worker data generator
├── featureExtraction.py         # Feature engineering module
├── inference.py                 # Inference and training utilities
├── pyproject.toml               # Python dependencies
├── behaviour_analysis.ipynb     # Behavioral pattern analysis notebook
├── prophetLSTM_training.ipynb   # Model training notebook
├── models/                      # Trained model artifacts
│   ├── feature_pipeline.joblib
│   ├── lstm_residual_model.keras
│   └── serialized_model.json
├── uploads/                     # Data storage
│   ├── input.csv
│   ├── features.csv
│   └── predictions.csv
└── worker_*.csv                 # Sample worker datasets
```

## Installation

### Prerequisites

- Python 3.12+
- pip or uv package manager

### Dependencies

```bash
pip install -r requirements.txt
```

Key packages:

- `prophet` - Time series forecasting
- `tensorflow` - Deep learning
- `scikit-learn` - Machine learning utilities
- `pandas` & `numpy` - Data manipulation
- `flask` - REST API server
- `matplotlib` & `seaborn` - Visualization

## Usage

### 1. Starting the API Server

```bash
python main.py
```

Server runs on `http://0.0.0.0:5000` by default (configurable via `PORT` environment variable).

### 2. Training the Model

**Endpoint**: `POST /train`

**Request Body**:

```json
[
  {
    "Date": "2024-01-01",
    "Category": "Transport",
    "Platform_Count": 2,
    "Daily_Expenses": 1500,
    "Daily_Income": 3000
  },
  ...
]
```

**Response**:

```json
{
  "status": "trained",
  "rows": 365,
  "train_rows": 328,
  "message": "Prophet model trained and 7-day forecast prepared.",
  "csv_path": "uploads/predictions.csv",
  "accuracy": {
    "train": {
      "mae": 0.1234,
      "rmse": 0.2345,
      "r2": 0.8765
    },
    "validation": {
      "mae": 0.1456,
      "rmse": 0.2567,
      "r2": 0.8543
    }
  }
}
```

### 3. Getting Predictions

**7-Day Forecast**:

```bash
GET /predict_7_days
```

**Growth Trend**:

```bash
GET /growth_trend
```

**Seasonality Analysis**:

```bash
GET /seasonality
```

## Data Generation

The `dataGenerator.py` script creates synthetic datasets for three worker archetypes:

### Worker Types

1. **Worker A - Driver (Uber/Ola/Swiggy)**

   - High frequency, consistent income
   - Weekend peaks
   - High fuel/maintenance costs (~35% of income)

2. **Worker B - Freelancer (Tech/Design)**

   - Low frequency, high variance
   - Invoice-based payments (monthly spikes)
   - Low variable expenses, subscription costs

3. **Worker C - Hybrid (Student/Part-time)**
   - Random mix of delivery and micro-tasks
   - Irregular income patterns
   - Variable expense patterns

### Running Data Generator

```bash
python dataGenerator.py
```

Generates:

- `worker_a_driver.csv`
- `worker_b_freelancer.csv`
- `worker_c_hybrid.csv`

## Feature Engineering

The `featureExtraction.py` module creates derived features:

- **Income Volatility Index**: 7-day rolling standard deviation
- **Income Rolling Average**: 90-day moving average
- **Expense-to-Income Ratio**: Financial health indicator
- **Net Income**: Daily profit/loss
- **Savings**: Monthly surplus/deficit
- **Standardized Target**: Normalized income for model training

## Model Training

### Prophet Component

- Handles seasonal decomposition
- Weekly and yearly seasonality
- Holiday effects
- Trend estimation

### LSTM Component

- 32-unit LSTM layer
- 16-unit dense layer with ReLU activation
- Output layer for residual prediction
- Early stopping with patience=10
- MSE loss optimization

### Training Configuration

```python
seq_len = 45         # Sequence length for LSTM
epochs = 100         # Maximum training epochs
batch_size = 32      # Mini-batch size
train_split = 0.9    # 90% training, 10% validation
```

## API Integration

### External Data Sources

- **Fuel Prices**: Indian API for petrol prices (Mumbai)
- **Unemployment Rates**: Monthly economic indicators

### Environment Variables

```bash
PETROL_API=<your_api_key>
PORT=5000
```

## Data Flow

1. **Input** → Raw transaction JSON
2. **Preprocessing** → Feature extraction, standardization
3. **Training** → Prophet fit + feature pipeline creation
4. **Prediction** → 7-day forecast generation
5. **Output** → Predictions saved to CSV and returned via API

## Files Generated

- `uploads/input.csv` - Raw input data
- `uploads/features.csv` - Engineered features
- `uploads/predictions.csv` - 7-day forecasts with components
- `models/feature_pipeline.joblib` - Trained preprocessing pipeline
- `models/lstm_residual_model.keras` - Trained LSTM model

## Performance Metrics

The system provides comprehensive accuracy metrics:

- **MAE** (Mean Absolute Error): Average prediction error
- **RMSE** (Root Mean Squared Error): Penalized large errors
- **R²** (Coefficient of Determination): Variance explained

Metrics are computed for both training and validation sets.

## Notebooks

### `behaviour_analysis.ipynb`

- Exploratory data analysis
- Worker behavior pattern identification
- Statistical insights

### `prophetLSTM_training.ipynb`

- Interactive model training
- Hyperparameter tuning
- Visualization of results

## Limitations & Considerations

- Requires minimum 45 days of historical data for optimal LSTM performance
- External API dependencies (fuel prices) may require fallback values
- Model assumes continuation of historical patterns
- Indian market context (holidays, economic indicators)

## Future Enhancements

- [ ] Multi-horizon forecasting (14, 30 days)
- [ ] Anomaly detection for unusual spending patterns
- [ ] Personalized recommendations based on predictions
- [ ] Integration with real-time transaction streaming
- [ ] Support for additional external economic indicators

## Contributing

When adding new features:

1. Update feature extraction in `featureExtraction.py`
2. Retrain the pipeline and models
3. Update API documentation
4. Add tests for new endpoints

## License

See main project LICENSE file.

## Support

For issues or questions, refer to the main project documentation or raise an issue in the repository.
