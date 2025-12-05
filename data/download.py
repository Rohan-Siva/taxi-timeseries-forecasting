import os
import requests
from tqdm import tqdm
import pandas as pd


def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)


def download_taxi_data(year=2023, months=range(1, 7)):
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    os.makedirs('raw', exist_ok=True)
    
    for month in months:
        filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
        url = f"{base_url}/{filename}"
        filepath = os.path.join('raw', filename)
        
        if os.path.exists(filepath):
            print(f"✓ {filename} already exists, skipping...")
            continue
            
        print(f"Downloading {filename}...")
        try:
            download_file(url, filepath)
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")


def combine_data(year=2023, months=range(1, 7)):
    dfs = []
    
    for month in months:
        filename = f"yellow_tripdata_{year}-{month:02d}.parquet"
        filepath = os.path.join('raw', filename)
        
        if os.path.exists(filepath):
            print(f"Reading {filename}...")
            df = pd.read_parquet(filepath)
            dfs.append(df)
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        output_path = 'raw/combined_taxi_data.csv'
        print(f"Saving combined data to {output_path}...")
        combined_df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(combined_df):,} records")
        return combined_df
    else:
        print("No data files found!")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("NYC Taxi Data Downloader")
    print("=" * 60)
    
                                              
    download_taxi_data(year=2023, months=range(1, 7))
    
                              
    print("\n" + "=" * 60)
    print("Combining data files...")
    print("=" * 60)
    combine_data(year=2023, months=range(1, 7))
    
    print("\n✓ Download complete!")
