"""
Evaluation Metrics for Time Series Forecasting
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
                            
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denominator != 0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_forecast(y_true, y_pred, model_name="Model"):
  
    metrics = {
        'model': model_name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': root_mean_squared_error(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'SMAPE': symmetric_mean_absolute_percentage_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    
    return metrics


def evaluate_all_models(predictions_dict, y_true):
    
    results = []
    
    for model_name, y_pred in predictions_dict.items():
                            
        min_len = min(len(y_true), len(y_pred))
        metrics = evaluate_forecast(
            y_true[:min_len],
            y_pred[:min_len],
            model_name=model_name
        )
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('RMSE')
    
    return results_df


def print_metrics_table(metrics_df):
    print("\n" + "="*80)
    print("MODEL EVALUATION RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'MAE':>10} {'RMSE':>10} {'MAPE':>10} {'SMAPE':>10} {'R²':>10}")
    print("-"*80)
    
    for _, row in metrics_df.iterrows():
        print(f"{row['model']:<20} {row['MAE']:>10.2f} {row['RMSE']:>10.2f} "
              f"{row['MAPE']:>9.2f}% {row['SMAPE']:>9.2f}% {row['R2']:>10.4f}")
    
    print("="*80)


def save_metrics(metrics_df, output_path='results/evaluation_metrics.csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved metrics to {output_path}")


def main():
    print("="*60)
    print("Model Evaluation")
    print("="*60)
    
                    
    test = pd.read_csv('data/processed/test.csv', index_col='timestamp', parse_dates=True)
    y_true = test['trip_count'].values
    
                                      
    predictions = {}
    
                     
    try:
        baseline_preds = pd.read_csv('results/baseline_predictions.csv')
        if 'pred_arima' in baseline_preds.columns:
            predictions['ARIMA'] = baseline_preds['pred_arima'].values
        if 'pred_exp_smoothing' in baseline_preds.columns:
            predictions['Exp Smoothing'] = baseline_preds['pred_exp_smoothing'].values
        if 'pred_prophet' in baseline_preds.columns:
            predictions['Prophet'] = baseline_preds['pred_prophet'].values
    except FileNotFoundError:
        print("Warning: Baseline predictions not found")
    
                          
    try:
        lstm_preds = pd.read_csv('results/lstm_predictions.csv')
        predictions['LSTM'] = lstm_preds['pred_lstm'].values
    except FileNotFoundError:
        print("Warning: LSTM predictions not found")
    
    try:
        gru_preds = pd.read_csv('results/gru_predictions.csv')
        predictions['GRU'] = gru_preds['pred_gru'].values
    except FileNotFoundError:
        print("Warning: GRU predictions not found")
    
    try:
        transformer_preds = pd.read_csv('results/transformer_predictions.csv')
        predictions['Transformer'] = transformer_preds['pred_transformer'].values
    except FileNotFoundError:
        print("Warning: Transformer predictions not found")
    
    if not predictions:
        print("Error: No predictions found!")
        return
    
                         
    metrics_df = evaluate_all_models(predictions, y_true)
    
                   
    print_metrics_table(metrics_df)
    
                  
    save_metrics(metrics_df)
    
    print("\n" + "="*60)
    print("✓ Evaluation complete!")
    print("="*60)
    
    return metrics_df


if __name__ == "__main__":
    metrics_df = main()
