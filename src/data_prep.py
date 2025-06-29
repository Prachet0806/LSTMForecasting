#data_prep.py
# Data Preparation Script for Stock Price Prediction
# This script downloads stock data, processes it, and prepares it for training an LSTM model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import config
import os
import yfinance as yf
import joblib

TICKER = "AAPL"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
RAW_PATH = "data/raw"
os.makedirs(RAW_PATH, exist_ok=True)
raw_file = os.path.join(RAW_PATH, f"{TICKER}_stock_data.csv")

def download_data():
    """Download stock data if not present"""
    if not os.path.exists(raw_file):
        print("Downloading data...")
        data = yf.download(TICKER, start=START_DATE, end=END_DATE)
        if data is not None and not data.empty:
            data.to_csv(raw_file)
            print(f"Raw data saved to {raw_file}")
        else:
            print("Failed to download data")
    else:
        print(f"File already exists: {raw_file}")

def compute_technical_indicators(data):
    """Compute technical indicators for the data"""
    
    def compute_macd(close, fast=12, slow=26, signal=9):
        exp1 = close.ewm(span=fast, adjust=False).mean()
        exp2 = close.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line

    def compute_rsi(close, window=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_ema(close, periods=[12, 26]):
        ema_features = {}
        for period in periods:
            ema_features[f'EMA_{period}'] = close.ewm(span=period, adjust=False).mean()
        return ema_features

    def compute_bollinger_bands(close, window=20, num_std=2):
        sma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        bb_position = (close - lower_band) / (upper_band - lower_band)  # BB position (0-1)
        return upper_band, lower_band, bb_position

    # Compute indicators
    macd, macd_signal = compute_macd(data['Close'])
    rsi = compute_rsi(data['Close'])
    ema_features = compute_ema(data['Close'])
    bb_upper, bb_lower, bb_position = compute_bollinger_bands(data['Close'])

    # Add indicators to data
    data['MACD'] = macd
    data['RSI'] = rsi
    data['EMA_12'] = ema_features['EMA_12']
    data['EMA_26'] = ema_features['EMA_26']
    data['BB_upper'] = bb_upper
    data['BB_lower'] = bb_lower
    data['BB_position'] = bb_position

    return data

def prepare_data(seq_length=None):
    """Prepare data with specified sequence length"""
    if seq_length is None:
        seq_length = config.SEQ_LENGTH
    
    # Download data if needed
    download_data()
    
    # Load and preprocess raw data
    raw_file = "data/raw/AAPL_stock_data.csv"
    data = pd.read_csv(raw_file, skiprows=2)  # data is a pandas DataFrame

    # Rename columns properly
    data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

    # Convert Date and set as index
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

    # Select relevant columns and convert to numeric
    feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
    data = data[feature_cols]
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Add technical indicators
    data = compute_technical_indicators(data)

    # Final feature columns
    feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 'MACD', 'RSI', 'EMA_12', 'EMA_26', 'BB_upper', 'BB_lower', 'BB_position']
    data = data[feature_cols]
    data = data.dropna()  # This is a pandas DataFrame, so dropna() is valid

    # Create sequences for all features, target is Close
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data.iloc[i-seq_length:i].values)
        y.append(data.iloc[i, 0])  # Close is the first column
    X, y = np.array(X), np.array(y)

    # Train-test split
    split = int(config.TEST_SPLIT * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Fit scaler only on training data (all features)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_flat = X_train.reshape(-1, X_train.shape[2])
    scaler.fit(X_train_flat)

    # Scale all sets
    X_train_scaled = scaler.transform(X_train.reshape(-1, X_train.shape[2])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    # Scale y (target) using only Close column
    close_scaler = MinMaxScaler(feature_range=(0, 1))
    close_scaler.fit(y_train.reshape(-1, 1))
    y_train_scaled = close_scaler.transform(y_train.reshape(-1, 1)).reshape(-1)
    y_test_scaled = close_scaler.transform(y_test.reshape(-1, 1)).reshape(-1)

    # Save processed data and scalers
    np.save('data/processed/X_train.npy', X_train_scaled)
    np.save('data/processed/X_test.npy', X_test_scaled)
    np.save('data/processed/y_train.npy', y_train_scaled)
    np.save('data/processed/y_test.npy', y_test_scaled)
    joblib.dump(scaler, 'data/processed/X_scaler.pkl')
    joblib.dump(close_scaler, 'data/processed/y_scaler.pkl')

    print(f"Data prepared with sequence length {seq_length}")
    print(f"Train samples: {X_train_scaled.shape}, Test samples: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler, close_scaler

if __name__ == "__main__":
    # Use default sequence length from config
    prepare_data()
