# NYC Taxi Demand Forecasting & Anomaly Detection

## Project Overview
This project implements an end-to-end machine learning pipeline for forecasting NYC taxi demand and detecting anomalies. It leverages a variety of models ranging from statistical baselines to advanced deep learning architectures.

**Key Features:**
- **Time Series Forecasting:** Predicting hourly taxi trip counts.
- **Anomaly Detection:** Identifying unusual spikes or drops in demand (e.g., holidays, extreme weather).
- **Comprehensive Evaluation:** rigorous testing with walk-forward cross-validation.

## Tech Stack
- **Language:** Python 3.10+
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Statistical Models:** Statsmodels (ARIMA, Exponential Smoothing), Prophet
- **Deep Learning:** PyTorch (LSTM, GRU, Transformer)
- **Anomaly Detection:** PyOD, Scikit-learn (Isolation Forest), PyTorch (VAE, LSTM-AE)

## Project Structure
```
taxi-timeseries-forecasting/
├── data/
│   ├── download.py          # Script to download NYC TLC data
│   └── preprocess.py        # Data cleaning, aggregation, and feature engineering
├── models/
│   ├── baseline.py          # ARIMA, Prophet, Exponential Smoothing
│   ├── lstm_model.py        # LSTM implementation
│   ├── gru_model.py         # GRU implementation
│   ├── transformer_model.py # Transformer implementation
│   └── anomaly_detection.py # VAE, Isolation Forest, LSTM-AE
├── evaluation/
│   ├── metrics.py           # MAE, RMSE, MAPE, SMAPE implementations
│   ├── cross_validation.py  # Walk-forward validation
│   └── visualization.py     # Plotting utilities
├── notebooks/
│   └── eda_exploration.ipynb # Exploratory Data Analysis
├── results/                 # Saved models, predictions, and plots
├── main.py                  # Main execution script
└── requirements.txt         # Project dependencies
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd taxi-timeseries-forecasting
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the pipeline:**
   To download data, train all models, and generate results:
   ```bash
   python main.py --download --epochs 50
   ```
   
   If you already have the data:
   ```bash
   python main.py --epochs 50
   ```

## Model Architectures

### Forecasting Models
1. **ARIMA:** Statistical baseline for capturing linear trends and seasonality.
2. **Prophet:** Additive regression model handling daily/weekly seasonality and holidays.
3. **LSTM (Long Short-Term Memory):** RNN variant capable of learning long-term dependencies.
4. **GRU (Gated Recurrent Unit):** Simplified RNN, often faster and as effective as LSTM.
5. **Transformer:** Attention-based architecture for capturing complex global dependencies.

### Anomaly Detection
1. **Isolation Forest:** Tree-based ensemble method for outlier detection.
2. **VAE (Variational Autoencoder):** Probabilistic reconstruction-based detection.
3. **LSTM Autoencoder:** Reconstruction-based detection using temporal sequences.

## Results & Evaluation

The models are evaluated using **RMSE** (Root Mean Squared Error) and **MAPE** (Mean Absolute Percentage Error) on a hold-out test set using walk-forward validation.

### Performance Comparison (Example)
| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| ARIMA | 1245.3 | 1560.2 | 12.5% |
| Prophet | 1100.5 | 1350.8 | 10.2% |
| LSTM | 850.2 | 1050.4 | 8.1% |
| Transformer | **820.1** | **980.5** | **7.8%** |

*(Note: Run `main.py` to generate the latest metrics for your specific data subset)*

## Visualizations
The pipeline automatically generates the following plots in `results/plots/`:
- **Time Series Decomposition:** Trend, Seasonality, Residuals.
- **Forecasts:** Comparison of model predictions vs actuals.
- **Anomalies:** Highlighted anomalies detected by each method.

## Future Improvements
- Incorporate weather data (temperature, precipitation) as exogenous variables.
- Implement hyperparameter tuning (Optuna/Ray Tune).
- Deploy model as a REST API using FastAPI.

## **Contact**
For collaborations or questions, please reach out to rohansiva123@gmail.com.
