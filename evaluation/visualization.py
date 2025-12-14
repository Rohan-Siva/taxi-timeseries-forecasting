import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 6)
plt.rcParams['font.size'] = 10

def plot_time_series(data, title='Time Series', output_path=None):
    plt.figure(figsize=(15, 6))
    plt.plot(data.index, data['trip_count'], linewidth=1, alpha=0.8)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Trip Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved plot to {output_path}')
    else:
        plt.show()
    plt.close()

def plot_decomposition(decomposition, output_path=None):
    fig, axes = plt.subplots(4, 1, figsize=(15, 10))
    decomposition.observed.plot(ax=axes[0], title='Observed', color='#2E86AB')
    axes[0].set_ylabel('Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend', color='#A23B72')
    axes[1].set_ylabel('Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal', color='#F18F01')
    axes[2].set_ylabel('Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual', color='#C73E1D')
    axes[3].set_ylabel('Residual')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved decomposition plot to {output_path}')
    else:
        plt.show()
    plt.close()

def plot_predictions_vs_actual(actual, predictions_dict, title='Predictions vs Actual', output_path=None, n_samples=500):
    plt.figure(figsize=(15, 8))
    if len(actual) > n_samples:
        actual = actual[-n_samples:]
        predictions_dict = {k: v[-n_samples:] for k, v in predictions_dict.items()}
    plt.plot(actual.index, actual.values, label='Actual', linewidth=2, color='black', alpha=0.7)
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51']
    for i, (model_name, preds) in enumerate(predictions_dict.items()):
        plt.plot(actual.index, preds, label=model_name, linewidth=1.5, alpha=0.7, color=colors[i % len(colors)])
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Trip Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved predictions plot to {output_path}')
    else:
        plt.show()
    plt.close()

def plot_model_comparison(metrics_df, output_path=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    metrics = ['MAE', 'RMSE', 'MAPE', 'R2']
    titles = ['Mean Absolute Error', 'Root Mean Squared Error', 'Mean Absolute Percentage Error (%)', 'R² Score']
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        sorted_df = metrics_df.sort_values(metric, ascending=metric != 'R2')
        bars = ax.barh(sorted_df['model'], sorted_df[metric])
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved model comparison to {output_path}')
    else:
        plt.show()
    plt.close()

def plot_anomalies(data, anomaly_scores, method_name='Anomaly Detection', threshold_percentile=95, output_path=None, n_samples=1000):
    plt.figure(figsize=(15, 8))
    if len(data) > n_samples:
        data = data[-n_samples:]
        anomaly_scores = anomaly_scores[-n_samples:]
    threshold = np.percentile(anomaly_scores[anomaly_scores > 0], threshold_percentile)
    anomalies = anomaly_scores > threshold
    plt.plot(data.index, data['trip_count'].values, linewidth=1, alpha=0.7, color='#2E86AB', label='Normal')
    anomaly_points = data[anomalies]
    plt.scatter(anomaly_points.index, anomaly_points['trip_count'].values, color='red', s=50, alpha=0.7, label='Anomaly', zorder=5)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Trip Count', fontsize=12)
    plt.title(f'{method_name} - Detected Anomalies', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved anomaly plot to {output_path}')
    else:
        plt.show()
    plt.close()
    return anomalies.sum()

def plot_anomaly_scores(data, anomaly_scores_dict, output_path=None, n_samples=1000):
    fig, axes = plt.subplots(len(anomaly_scores_dict), 1, figsize=(15, 4 * len(anomaly_scores_dict)))
    if len(anomaly_scores_dict) == 1:
        axes = [axes]
    if len(data) > n_samples:
        data = data[-n_samples:]
        anomaly_scores_dict = {k: v[-n_samples:] for k, v in anomaly_scores_dict.items()}
    for idx, (method, scores) in enumerate(anomaly_scores_dict.items()):
        ax = axes[idx]
        ax.plot(data.index, scores, linewidth=1, alpha=0.8, color='#2E86AB')
        threshold = np.percentile(scores[scores > 0], 95)
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'95th percentile: {threshold:.4f}')
        ax.set_ylabel('Anomaly Score', fontsize=11)
        ax.set_title(f"{method.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time', fontsize=12)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'✓ Saved anomaly scores plot to {output_path}')
    else:
        plt.show()
    plt.close()

def create_all_visualizations():
    print('=' * 60)
    print('Creating Visualizations')
    print('=' * 60)
    output_dir = 'results/plots'
    os.makedirs(output_dir, exist_ok=True)
    print('\n1. Plotting time series...')
    data = pd.read_csv('data/processed/hourly_demand.csv', index_col='timestamp', parse_dates=True)
    plot_time_series(data, title='NYC Taxi Hourly Demand', output_path=os.path.join(output_dir, 'time_series.png'))
    print('\n2. Plotting decomposition...')
    from statsmodels.tsa.seasonal import seasonal_decompose
    decomposition = seasonal_decompose(data['trip_count'], model='additive', period=24)
    plot_decomposition(decomposition, output_path=os.path.join(output_dir, 'decomposition.png'))
    print('\n3. Plotting predictions...')
    test = pd.read_csv('data/processed/test.csv', index_col='timestamp', parse_dates=True)
    predictions = {}
    try:
        baseline_preds = pd.read_csv('results/baseline_predictions.csv', index_col='timestamp', parse_dates=True)
        if 'pred_arima' in baseline_preds.columns:
            predictions['ARIMA'] = baseline_preds['pred_arima'].values
        if 'pred_prophet' in baseline_preds.columns:
            predictions['Prophet'] = baseline_preds['pred_prophet'].values
    except:
        pass
    try:
        lstm_preds = pd.read_csv('results/lstm_predictions.csv', index_col='timestamp', parse_dates=True)
        predictions['LSTM'] = lstm_preds['pred_lstm'].values
    except:
        pass
    try:
        gru_preds = pd.read_csv('results/gru_predictions.csv', index_col='timestamp', parse_dates=True)
        predictions['GRU'] = gru_preds['pred_gru'].values
    except:
        pass
    try:
        transformer_preds = pd.read_csv('results/transformer_predictions.csv', index_col='timestamp', parse_dates=True)
        predictions['Transformer'] = transformer_preds['pred_transformer'].values
    except:
        pass
    if predictions:
        plot_predictions_vs_actual(test['trip_count'], predictions, title='Model Predictions vs Actual Values', output_path=os.path.join(output_dir, 'predictions_comparison.png'))
    print('\n4. Plotting model comparison...')
    try:
        metrics_df = pd.read_csv('results/evaluation_metrics.csv')
        plot_model_comparison(metrics_df, output_path=os.path.join(output_dir, 'model_comparison.png'))
    except:
        print('  Metrics not found, skipping...')
    print('\n5. Plotting anomalies...')
    try:
        anomaly_df = pd.read_csv('results/anomaly_scores.csv', index_col='timestamp', parse_dates=True)
        anomaly_methods = {'Isolation Forest': 'anomaly_score_isolation_forest', 'VAE': 'anomaly_score_vae', 'LSTM Reconstruction': 'anomaly_score_lstm_reconstruction'}
        for method_name, col_name in anomaly_methods.items():
            if col_name in anomaly_df.columns:
                n_anomalies = plot_anomalies(anomaly_df, anomaly_df[col_name].values, method_name=method_name, output_path=os.path.join(output_dir, f"anomalies_{col_name.split('_')[-1]}.png"))
                print(f'  {method_name}: {n_anomalies} anomalies detected')
        scores_dict = {k: anomaly_df[v].values for k, v in anomaly_methods.items() if v in anomaly_df.columns}
        if scores_dict:
            plot_anomaly_scores(anomaly_df, scores_dict, output_path=os.path.join(output_dir, 'anomaly_scores.png'))
    except Exception as e:
        print(f'  Error plotting anomalies: {e}')
    print('\n' + '=' * 60)
    print('✓ All visualizations created!')
    print('=' * 60)
if __name__ == '__main__':
    create_all_visualizations()