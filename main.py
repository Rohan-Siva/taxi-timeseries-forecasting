import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from data.download import download_taxi_data, combine_data
from data.preprocess import TaxiDataPreprocessor
from models.baseline import BaselineModels
from models.lstm_model import main as train_lstm
from models.gru_model import main as train_gru
from models.transformer_model import main as train_transformer
from models.anomaly_detection import AnomalyDetector
from evaluation.metrics import evaluate_all_models, print_metrics_table, save_metrics
from evaluation.visualization import create_all_visualizations

def run_pipeline(download=False, epochs=20):
    print('=' * 80)
    print('NYC TAXI DEMAND FORECASTING & ANOMALY DETECTION')
    print('=' * 80)
    if download:
        print('\n[Step 1] Downloading Data...')
        download_taxi_data(year=2023, months=range(1, 7))
        combine_data(year=2023, months=range(1, 7))
    print('\n[Step 2] Preprocessing Data...')
    if not os.path.exists('raw/combined_taxi_data.csv') and (not download):
        print('Data not found! Please run with --download flag first.')
        return
    preprocessor = TaxiDataPreprocessor()
    preprocessor.load_data(sample_frac=0.1)
    preprocessor.aggregate_to_hourly()
    preprocessor.check_stationarity()
    preprocessor.decompose_series()
    preprocessor.add_time_features()
    preprocessor.normalize_data()
    preprocessor.save_processed_data()
    print('\n[Step 3] Training Baseline Models...')
    train = pd.read_csv('data/processed/train.csv', index_col='timestamp', parse_dates=True)
    test = pd.read_csv('data/processed/test.csv', index_col='timestamp', parse_dates=True)
    baseline = BaselineModels(train, test)
    baseline.fit_arima()
    baseline.fit_exponential_smoothing()
    baseline.fit_prophet()
    baseline.save_predictions()
    print('\n[Step 4] Training Deep Learning Models...')
    print('\n  > Training LSTM...')
    try:
        train_lstm()
    except Exception as e:
        print(f'Error training LSTM: {e}')
    try:
        train_gru()
    except Exception as e:
        print(f'Error training GRU: {e}')
    try:
        train_transformer()
    except Exception as e:
        print(f'Error training Transformer: {e}')
    print('\n[Step 5] Running Anomaly Detection...')
    full_data = pd.read_csv('data/processed/hourly_demand.csv', index_col='timestamp', parse_dates=True)
    detector = AnomalyDetector(full_data)
    detector.detect_with_isolation_forest()
    detector.detect_with_vae(epochs=epochs)
    detector.detect_with_lstm_reconstruction(epochs=epochs)
    detector.save_anomaly_scores()
    print('\n[Step 6] Evaluation & Visualization...')
    predictions = {}
    try:
        base_preds = pd.read_csv('results/baseline_predictions.csv')
        if 'pred_arima' in base_preds.columns:
            predictions['ARIMA'] = base_preds['pred_arima'].values
        if 'pred_prophet' in base_preds.columns:
            predictions['Prophet'] = base_preds['pred_prophet'].values
    except:
        pass
    try:
        predictions['LSTM'] = pd.read_csv('results/lstm_predictions.csv')['pred_lstm'].values
    except:
        pass
    try:
        predictions['GRU'] = pd.read_csv('results/gru_predictions.csv')['pred_gru'].values
    except:
        pass
    try:
        predictions['Transformer'] = pd.read_csv('results/transformer_predictions.csv')['pred_transformer'].values
    except:
        pass
    if predictions:
        y_true = pd.read_csv('data/processed/test.csv')['trip_count'].values
        metrics_df = evaluate_all_models(predictions, y_true)
        print_metrics_table(metrics_df)
        save_metrics(metrics_df)
    create_all_visualizations()
    print('\n' + '=' * 80)
    print('PROJECT COMPLETED SUCCESSFULLY')
    print('=' * 80)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Taxi Demand Forecasting Pipeline')
    parser.add_argument('--download', action='store_true', help='Download fresh data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for DL models')
    args = parser.parse_args()
    run_pipeline(download=args.download, epochs=args.epochs)