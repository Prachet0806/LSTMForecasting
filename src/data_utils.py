#data_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_and_scale(filepath, feature="Close"):
    df = pd.read_csv(filepath)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[[feature]])
    return scaled, scaler

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def load_processed_data():
    X_train = np.load('data/processed/X_train.npy')
    X_test = np.load('data/processed/X_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_test = np.load('data/processed/y_test.npy')
    X_scaler = joblib.load('data/processed/X_scaler.pkl')
    y_scaler = joblib.load('data/processed/y_scaler.pkl')
    return X_train, X_test, y_train, y_test, X_scaler, y_scaler

def get_num_features():
    X_train = np.load('data/processed/X_train.npy')
    return X_train.shape[2]
