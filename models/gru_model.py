import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
import os

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

class GRUForecaster(nn.Module):

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        prediction = self.fc(last_output)
        return prediction

class GRUTrainer:

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
        print(f'\nTraining GRU for {epochs} epochs...')
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

    def save_model(self, path='models/saved/gru_model.pt'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict(), 'train_losses': self.train_losses, 'val_losses': self.val_losses}, path)
        print(f'✓ Model saved to {path}')

def main():
    print('=' * 60)
    print('GRU Time Series Forecasting')
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
    model = GRUForecaster(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    trainer = GRUTrainer(model, device=device)
    trainer.fit(train_loader, test_loader, epochs=50, lr=0.001)
    trainer.save_model()
    predictions_norm = trainer.predict(test_loader)
    predictions = predictions_norm * norm_params['std'] + norm_params['mean']
    os.makedirs('results', exist_ok=True)
    pred_df = pd.DataFrame({'timestamp': test.index[seq_length:], 'actual': test['trip_count'].values[seq_length:], 'pred_gru': predictions})
    pred_df.to_csv('results/gru_predictions.csv', index=False)
    print(f'✓ Saved predictions to results/gru_predictions.csv')
    print('\n' + '=' * 60)
    print('✓ GRU training complete!')
    print('=' * 60)
    return trainer
if __name__ == '__main__':
    trainer = main()