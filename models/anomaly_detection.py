import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import json
import os

class VAE(nn.Module):

    def __init__(self, input_dim=24, hidden_dim=16, latent_dim=8):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU())
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, x):
        h = self.encoder(x)
        return (self.fc_mu(h), self.fc_logvar(h))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return (self.decode(z), mu, logvar)

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.001 * kld

class TimeSeriesDatasetVAE(Dataset):

    def __init__(self, data, seq_length=24, target_col='trip_count'):
        self.data = data[target_col].values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length + 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        return torch.FloatTensor(x)

class LSTMAutoencoder(nn.Module):

    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=input_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, cell) = self.encoder(x)
        seq_len = x.size(1)
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        reconstructed, _ = self.decoder(hidden_repeated)
        return reconstructed

class AnomalyDetector:

    def __init__(self, data, seq_length=24):
        self.data = data
        self.seq_length = seq_length
        self.anomaly_scores = {}

    def detect_with_isolation_forest(self, contamination=0.05):
        print('\n' + '=' * 60)
        print('Isolation Forest Anomaly Detection')
        print('=' * 60)
        windows = []
        for i in range(len(self.data) - self.seq_length + 1):
            window = self.data['trip_count'].values[i:i + self.seq_length]
            windows.append(window)
        windows = np.array(windows)
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        predictions = iso_forest.fit_predict(windows)
        scores = iso_forest.score_samples(windows)
        anomalies = predictions == -1
        print(f'Detected {anomalies.sum()} anomalies ({anomalies.sum() / len(anomalies) * 100:.2f}%)')
        padded_scores = np.zeros(len(self.data))
        padded_scores[self.seq_length - 1:] = scores
        self.anomaly_scores['isolation_forest'] = -padded_scores
        return (anomalies, scores)

    def detect_with_vae(self, epochs=30, threshold_percentile=95):
        print('\n' + '=' * 60)
        print('VAE Anomaly Detection')
        print('=' * 60)
        with open('data/processed/norm_params.json', 'r') as f:
            norm_params = json.load(f)
        data_norm = self.data.copy()
        data_norm['trip_count'] = (self.data['trip_count'] - norm_params['mean']) / norm_params['std']
        dataset = TimeSeriesDatasetVAE(data_norm, seq_length=self.seq_length)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = VAE(input_dim=self.seq_length, hidden_dim=16, latent_dim=8).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print(f'Training VAE for {epochs} epochs...')
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss = vae_loss(recon, batch, mu, logvar)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss / len(dataloader):.4f}')
        model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for i in range(len(dataset)):
                x = dataset[i].unsqueeze(0).to(device)
                recon, _, _ = model(x)
                error = torch.mean((recon - x) ** 2).item()
                reconstruction_errors.append(error)
        reconstruction_errors = np.array(reconstruction_errors)
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        anomalies = reconstruction_errors > threshold
        print(f'Detected {anomalies.sum()} anomalies ({anomalies.sum() / len(anomalies) * 100:.2f}%)')
        print(f'Threshold: {threshold:.4f}')
        padded_scores = np.zeros(len(self.data))
        padded_scores[self.seq_length - 1:] = reconstruction_errors
        self.anomaly_scores['vae'] = padded_scores
        return (anomalies, reconstruction_errors)

    def detect_with_lstm_reconstruction(self, epochs=30, threshold_percentile=95):
        print('\n' + '=' * 60)
        print('LSTM Reconstruction Anomaly Detection')
        print('=' * 60)
        with open('data/processed/norm_params.json', 'r') as f:
            norm_params = json.load(f)
        data_norm = self.data.copy()
        data_norm['trip_count'] = (self.data['trip_count'] - norm_params['mean']) / norm_params['std']
        dataset = TimeSeriesDatasetVAE(data_norm, seq_length=self.seq_length)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMAutoencoder(input_size=1, hidden_size=32, num_layers=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        print(f'Training LSTM Autoencoder for {epochs} epochs...')
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                batch = batch.unsqueeze(-1).to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}] - Loss: {total_loss / len(dataloader):.4f}')
        model.eval()
        reconstruction_errors = []
        with torch.no_grad():
            for i in range(len(dataset)):
                x = dataset[i].unsqueeze(0).unsqueeze(-1).to(device)
                recon = model(x)
                error = torch.mean((recon - x) ** 2).item()
                reconstruction_errors.append(error)
        reconstruction_errors = np.array(reconstruction_errors)
        threshold = np.percentile(reconstruction_errors, threshold_percentile)
        anomalies = reconstruction_errors > threshold
        print(f'Detected {anomalies.sum()} anomalies ({anomalies.sum() / len(anomalies) * 100:.2f}%)')
        print(f'Threshold: {threshold:.4f}')
        padded_scores = np.zeros(len(self.data))
        padded_scores[self.seq_length - 1:] = reconstruction_errors
        self.anomaly_scores['lstm_reconstruction'] = padded_scores
        return (anomalies, reconstruction_errors)

    def save_anomaly_scores(self, output_dir='results'):
        os.makedirs(output_dir, exist_ok=True)
        anomaly_df = pd.DataFrame({'timestamp': self.data.index, 'trip_count': self.data['trip_count'].values})
        for method, scores in self.anomaly_scores.items():
            anomaly_df[f'anomaly_score_{method}'] = scores
        output_path = os.path.join(output_dir, 'anomaly_scores.csv')
        anomaly_df.to_csv(output_path, index=False)
        print(f'\n✓ Saved anomaly scores to {output_path}')
        return anomaly_df

def main():
    print('=' * 60)
    print('Time Series Anomaly Detection')
    print('=' * 60)
    data = pd.read_csv('data/processed/hourly_demand.csv', index_col='timestamp', parse_dates=True)
    print(f'Data: {len(data)} samples')
    detector = AnomalyDetector(data, seq_length=24)
    detector.detect_with_isolation_forest(contamination=0.05)
    detector.detect_with_vae(epochs=30)
    detector.detect_with_lstm_reconstruction(epochs=30)
    detector.save_anomaly_scores()
    print('\n' + '=' * 60)
    print('✓ Anomaly detection complete!')
    print('=' * 60)
    return detector
if __name__ == '__main__':
    detector = main()