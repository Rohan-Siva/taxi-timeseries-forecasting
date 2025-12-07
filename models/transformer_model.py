import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
import os
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TimeSeriesDataset(Dataset):

    def __init__(self, data, seq_length=24, target_col='trip_count'):
        self.data = data[target_col].values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length]
        return (torch.FloatTensor(x), torch.FloatTensor([y]))

class TransformerForecaster(nn.Module):

    def __init__(self, input_size=1, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerForecaster, self).__init__()
        self.d_model = d_model
        self.input_embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        transformer_out = self.transformer_encoder(x)
        last_output = transformer_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction

class TransformerTrainer:

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.unsqueeze(-1).to(self.device)
            batch_y = batch_y.to(self.device)
            predictions = self.model(batch_x)
            loss = criterion(predictions, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)

    def validate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.unsqueeze(-1).to(self.device)
                batch_y = batch_y.to(self.device)
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader, epochs=50, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(f'\nTraining Transformer for {epochs} epochs...')
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        print('✓ Training complete!')

    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.unsqueeze(-1).to(self.device)
                pred = self.model(batch_x)
                predictions.extend(pred.cpu().numpy())
        return np.array(predictions).flatten()

    def save_model(self, path='models/saved/transformer_model.pt'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict(), 'train_losses': self.train_losses, 'val_losses': self.val_losses}, path)
        print(f'✓ Model saved to {path}')

def main():
    print('=' * 60)
    print('Transformer Time Series Forecasting')
    print('=' * 60)
    train = pd.read_csv('data/processed/train.csv', index_col='timestamp', parse_dates=True)
    test = pd.read_csv('data/processed/test.csv', index_col='timestamp', parse_dates=True)
    with open('data/processed/norm_params.json', 'r') as f:
        norm_params = json.load(f)
    train_norm = train.copy()
    test_norm = test.copy()
    train_norm['trip_count'] = (train['trip_count'] - norm_params['mean']) / norm_params['std']
    test_norm['trip_count'] = (test['trip_count'] - norm_params['mean']) / norm_params['std']
    seq_length = 24
    train_dataset = TimeSeriesDataset(train_norm, seq_length=seq_length)
    test_dataset = TimeSeriesDataset(test_norm, seq_length=seq_length)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    print(f'Train samples: {len(train_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = TransformerForecaster(input_size=1, d_model=64, nhead=4, num_layers=2, dropout=0.1)
    trainer = TransformerTrainer(model, device=device)
    trainer.fit(train_loader, test_loader, epochs=50, lr=0.001)
    trainer.save_model()
    predictions_norm = trainer.predict(test_loader)
    predictions = predictions_norm * norm_params['std'] + norm_params['mean']
    os.makedirs('results', exist_ok=True)
    pred_df = pd.DataFrame({'timestamp': test.index[seq_length:], 'actual': test['trip_count'].values[seq_length:], 'pred_transformer': predictions})
    pred_df.to_csv('results/transformer_predictions.csv', index=False)
    print(f'✓ Saved predictions to results/transformer_predictions.csv')
    print('\n' + '=' * 60)
    print('✓ Transformer training complete!')
    print('=' * 60)
    return trainer
if __name__ == '__main__':
    trainer = main()