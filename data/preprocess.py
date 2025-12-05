import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os


class TaxiDataPreprocessor:
    
    def __init__(self, filepath='raw/combined_taxi_data.csv'):
        self.filepath = filepath
        self.df = None
        self.hourly_demand = None
        
    def load_data(self, sample_frac=0.1):
        print(f"Loading data from {self.filepath}...")
        
                   
        self.df = pd.read_csv(self.filepath)
        print(f"Loaded {len(self.df):,} records")
        
                                                 
        if sample_frac < 1.0:
            self.df = self.df.sample(frac=sample_frac, random_state=42)
            print(f"Sampled to {len(self.df):,} records ({sample_frac*100}%)")
        
        return self.df
    
    def aggregate_to_hourly(self):
        print("\nAggregating to hourly demand...")
        
                                         
        self.df['tpep_pickup_datetime'] = pd.to_datetime(self.df['tpep_pickup_datetime'])
        
                      
        self.df['pickup_hour'] = self.df['tpep_pickup_datetime'].dt.floor('H')
        
                              
        self.hourly_demand = self.df.groupby('pickup_hour').size().reset_index(name='trip_count')
        self.hourly_demand.set_index('pickup_hour', inplace=True)
        
                                   
        full_range = pd.date_range(
            start=self.hourly_demand.index.min(),
            end=self.hourly_demand.index.max(),
            freq='H'
        )
        self.hourly_demand = self.hourly_demand.reindex(full_range, fill_value=0)
        self.hourly_demand.index.name = 'timestamp'
        
        print(f"Created hourly time series with {len(self.hourly_demand)} hours")
        print(f"Date range: {self.hourly_demand.index.min()} to {self.hourly_demand.index.max()}")
        
        return self.hourly_demand
    
    def check_stationarity(self, series=None):
        if series is None:
            series = self.hourly_demand['trip_count']
        
        print("\n" + "="*60)
        print("Stationarity Test (Augmented Dickey-Fuller)")
        print("="*60)
        
        result = adfuller(series.dropna())
        
        print(f"ADF Statistic: {result[0]:.6f}")
        print(f"p-value: {result[1]:.6f}")
        print(f"Critical Values:")
        for key, value in result[4].items():
            print(f"  {key}: {value:.3f}")
        
        if result[1] < 0.05:
            print("\n✓ Series is STATIONARY (p < 0.05)")
        else:
            print("\n✗ Series is NON-STATIONARY (p >= 0.05)")
            print("  Consider differencing or transformation")
        
        return result
    
    def decompose_series(self, period=24):
        print(f"\nDecomposing time series (period={period} hours)...")
        
        decomposition = seasonal_decompose(
            self.hourly_demand['trip_count'],
            model='additive',
            period=period
        )
        
        return decomposition
    
    def add_time_features(self):
        print("\nAdding time-based features...")
        
        self.hourly_demand['hour'] = self.hourly_demand.index.hour
        self.hourly_demand['day_of_week'] = self.hourly_demand.index.dayofweek
        self.hourly_demand['day_of_month'] = self.hourly_demand.index.day
        self.hourly_demand['month'] = self.hourly_demand.index.month
        self.hourly_demand['is_weekend'] = (self.hourly_demand.index.dayofweek >= 5).astype(int)
        
                           
        self.hourly_demand['hour_sin'] = np.sin(2 * np.pi * self.hourly_demand['hour'] / 24)
        self.hourly_demand['hour_cos'] = np.cos(2 * np.pi * self.hourly_demand['hour'] / 24)
        self.hourly_demand['day_sin'] = np.sin(2 * np.pi * self.hourly_demand['day_of_week'] / 7)
        self.hourly_demand['day_cos'] = np.cos(2 * np.pi * self.hourly_demand['day_of_week'] / 7)
        
        print(f"Added {len(self.hourly_demand.columns) - 1} time features")
        
        return self.hourly_demand
    
    def normalize_data(self):
        print("\nNormalizing data...")
        
        mean = self.hourly_demand['trip_count'].mean()
        std = self.hourly_demand['trip_count'].std()
        
        self.hourly_demand['trip_count_normalized'] = (
            (self.hourly_demand['trip_count'] - mean) / std
        )
        
                                       
        self.norm_params = {'mean': mean, 'std': std}
        
        print(f"Mean: {mean:.2f}, Std: {std:.2f}")
        
        return self.hourly_demand
    
    def create_train_test_split(self, test_size=0.2):
        print(f"\nCreating train/test split (test_size={test_size})...")
        
        split_idx = int(len(self.hourly_demand) * (1 - test_size))
        
        train = self.hourly_demand.iloc[:split_idx]
        test = self.hourly_demand.iloc[split_idx:]
        
        print(f"Train: {len(train)} hours ({train.index.min()} to {train.index.max()})")
        print(f"Test:  {len(test)} hours ({test.index.min()} to {test.index.max()})")
        
        return train, test
    
    def save_processed_data(self, output_dir='processed'):
        os.makedirs(output_dir, exist_ok=True)
        
                                  
        output_path = os.path.join(output_dir, 'hourly_demand.csv')
        self.hourly_demand.to_csv(output_path)
        print(f"\n✓ Saved processed data to {output_path}")
        
                                
        train, test = self.create_train_test_split()
        train.to_csv(os.path.join(output_dir, 'train.csv'))
        test.to_csv(os.path.join(output_dir, 'test.csv'))
        print(f"✓ Saved train/test splits")
        
                                       
        import json
        with open(os.path.join(output_dir, 'norm_params.json'), 'w') as f:
            json.dump(self.norm_params, f)
        print(f"✓ Saved normalization parameters")
        
        return output_path


def main():
    """Main preprocessing pipeline"""
    print("="*60)
    print("NYC Taxi Data Preprocessing Pipeline")
    print("="*60)
    
                             
    preprocessor = TaxiDataPreprocessor()
    
                                                                 
    preprocessor.load_data(sample_frac=1.0)
    
                         
    preprocessor.aggregate_to_hourly()
    
                        
    preprocessor.check_stationarity()
    
                      
    decomposition = preprocessor.decompose_series(period=24)
    
                  
    preprocessor.add_time_features()
    
               
    preprocessor.normalize_data()
    
                         
    preprocessor.save_processed_data()
    
    print("\n" + "="*60)
    print("✓ Preprocessing complete!")
    print("="*60)
    
    return preprocessor


if __name__ == "__main__":
    preprocessor = main()
