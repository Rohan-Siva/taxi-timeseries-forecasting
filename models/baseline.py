import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')
import json
import os

class BaselineModels:

    def __init__(self, train_data, test_data):
        self.train = train_data
        self.test = test_data
        self.predictions = {}

    def fit_arima(self, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24)):
        print(f'\nFitting ARIMA{order} with seasonal{seasonal_order}...')
        try:
            model = ARIMA(self.train['trip_count'], order=order, seasonal_order=seasonal_order)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=len(self.test))
            self.predictions['arima'] = forecast.values
            print(f'✓ ARIMA fitted successfully')
            print(f'  AIC: {fitted_model.aic:.2f}')
            print(f'  BIC: {fitted_model.bic:.2f}')
            return fitted_model
        except Exception as e:
            print(f'✗ ARIMA fitting failed: {e}')
            print('  Trying simpler ARIMA(1,1,1)...')
            model = ARIMA(self.train['trip_count'], order=(1, 1, 1))
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=len(self.test))
            self.predictions['arima'] = forecast.values
            return fitted_model

    def fit_exponential_smoothing(self, seasonal_periods=24):
        print(f'\nFitting Exponential Smoothing (seasonal_periods={seasonal_periods})...')
        try:
            model = ExponentialSmoothing(self.train['trip_count'], seasonal_periods=seasonal_periods, trend='add', seasonal='add')
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=len(self.test))
            self.predictions['exp_smoothing'] = forecast.values
            print(f'✓ Exponential Smoothing fitted successfully')
            return fitted_model
        except Exception as e:
            print(f'✗ Exponential Smoothing failed: {e}')
            return None

    def fit_prophet(self):
        print('\nFitting Prophet...')
        prophet_train = pd.DataFrame({'ds': self.train.index, 'y': self.train['trip_count'].values})
        model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False, seasonality_mode='additive')
        model.fit(prophet_train)
        future = pd.DataFrame({'ds': self.test.index})
        forecast = model.predict(future)
        self.predictions['prophet'] = forecast['yhat'].values
        print(f'✓ Prophet fitted successfully')
        return model

    def get_predictions(self):
        return self.predictions

    def save_predictions(self, output_dir='results'):
        os.makedirs(output_dir, exist_ok=True)
        pred_df = pd.DataFrame({'timestamp': self.test.index, 'actual': self.test['trip_count'].values})
        for model_name, preds in self.predictions.items():
            pred_df[f'pred_{model_name}'] = preds
        output_path = os.path.join(output_dir, 'baseline_predictions.csv')
        pred_df.to_csv(output_path, index=False)
        print(f'\n✓ Saved predictions to {output_path}')
        return pred_df

def main():
    print('=' * 60)
    print('Baseline Time Series Models')
    print('=' * 60)
    train = pd.read_csv('data/processed/train.csv', index_col='timestamp', parse_dates=True)
    test = pd.read_csv('data/processed/test.csv', index_col='timestamp', parse_dates=True)
    print(f'Train: {len(train)} samples')
    print(f'Test:  {len(test)} samples')
    baseline = BaselineModels(train, test)
    baseline.fit_arima()
    baseline.fit_exponential_smoothing()
    baseline.fit_prophet()
    baseline.save_predictions()
    print('\n' + '=' * 60)
    print('✓ Baseline models complete!')
    print('=' * 60)
    return baseline
if __name__ == '__main__':
    baseline = main()