import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

class TimeSeriesCV:

    def __init__(self, n_splits=5, test_size=None):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, data):
        n = len(data)
        if self.test_size is None:
            test_size = n // (self.n_splits + 1)
        else:
            test_size = self.test_size
        min_train = n // 2
        for i in range(self.n_splits):
            train_end = min_train + i * test_size
            test_end = train_end + test_size
            if test_end > n:
                break
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            yield (train_idx, test_idx)

    def get_n_splits(self):
        return self.n_splits

def walk_forward_validation(data, model_func, n_splits=5):
    cv = TimeSeriesCV(n_splits=n_splits)
    scores = []
    print(f'\nPerforming {n_splits}-fold walk-forward validation...')
    for fold, (train_idx, test_idx) in enumerate(cv.split(data), 1):
        train_data = data.iloc[train_idx]
        test_data = data.iloc[test_idx]
        predictions = model_func(train_data, test_data)
        rmse = np.sqrt(mean_squared_error(test_data['trip_count'], predictions))
        scores.append(rmse)
        print(f'Fold {fold}: RMSE = {rmse:.2f}')
    print(f'\nMean RMSE: {np.mean(scores):.2f} (+/- {np.std(scores):.2f})')
    return scores

def plot_cv_results(scores, model_name='Model', output_dir='results/plots'):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores) + 1), scores, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=np.mean(scores), color='r', linestyle='--', label=f'Mean: {np.mean(scores):.2f}')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('RMSE', fontsize=12)
    plt.title(f'{model_name} - Walk-Forward Cross-Validation', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, f'{model_name.lower()}_cv_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'✓ Saved CV plot to {output_path}')

def main():
    print('=' * 60)
    print('Time Series Cross-Validation')
    print('=' * 60)
    data = pd.read_csv('data/processed/hourly_demand.csv', index_col='timestamp', parse_dates=True)
    print(f'Data: {len(data)} samples')

    def moving_average_model(train, test, window=24):
        predictions = []
        for i in range(len(test)):
            if i == 0:
                recent_values = train['trip_count'].values[-window:]
            else:
                recent_values = np.concatenate([train['trip_count'].values[-window + i:], test['trip_count'].values[:i]])[-window:]
            pred = np.mean(recent_values)
            predictions.append(pred)
        return np.array(predictions)
    scores = walk_forward_validation(data, moving_average_model, n_splits=5)
    plot_cv_results(scores, model_name='Moving Average')
    print('\n' + '=' * 60)
    print('✓ Cross-validation complete!')
    print('=' * 60)
    return scores
if __name__ == '__main__':
    scores = main()