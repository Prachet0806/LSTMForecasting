#train.py
# This script trains an LSTM model on the preprocessed data.
# It loads the training data, defines the model, trains it, and saves the trained model
import torch
import torch.nn as nn
import numpy as np
import joblib
from model import LSTMModel
from data_utils import load_processed_data
from torch.utils.data import TensorDataset, DataLoader
import config
import matplotlib.pyplot as plt
import random
import os

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(
    input_size,
    hidden_size,
    num_layers,
    dropout,
    output_size,
    batch_size,
    lr,
    epochs,
    use_gpu=True,
    model_path=None,
    scaler_path=None,
    plot_loss_path=None,
    seed=42,
    use_huber_loss=None,
    num_heads=None,
    huber_delta=None
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    X_train, X_test, y_train, y_test, X_scaler, y_scaler = load_processed_data()
    
    # Use config defaults if not provided
    if use_huber_loss is None:
        use_huber_loss = config.USE_HUBER_LOSS
    if num_heads is None:
        num_heads = config.NUM_HEADS
    if huber_delta is None:
        huber_delta = config.HUBER_DELTA
    
    # Chronological split: use config-defined split ratio
    val_split = int(config.TRAIN_VAL_SPLIT * len(X_train))
    X_train_final = X_train[:val_split]
    X_val = X_train[val_split:]
    y_train_final = y_train[:val_split]
    y_val = y_train[val_split:]
    
    X_tensor = torch.tensor(X_train_final, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_final, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    
    train_dataset = TensorDataset(X_tensor, y_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = LSTMModel(
        input_size,
        hidden_size,
        num_layers,
        dropout=dropout,
        output_size=output_size,
        use_attention=True,
        num_heads=num_heads
    ).to(device)
    
    # Choose loss function
    if use_huber_loss:
        criterion = nn.HuberLoss(delta=huber_delta)
    else:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=config.LEARNING_RATE_PATIENCE, 
        factor=config.LEARNING_RATE_FACTOR
    )
    
    best_val_loss = float('inf')
    patience = config.EARLY_STOPPING_PATIENCE
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if model_path:
                torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    if scaler_path:
        joblib.dump(y_scaler, scaler_path)
    if plot_loss_path:
        plt.figure(figsize=(8,5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.tight_layout()
        plt.savefig(plot_loss_path)
        plt.close()
    
    return best_val_loss

def main():
    best_val_loss = train(
        config.INPUT_SIZE,
        config.HIDDEN_SIZE,
        config.NUM_LAYERS,
        config.DROPOUT,
        config.OUTPUT_SIZE,
        config.BATCH_SIZE,
        config.LR,
        config.EPOCHS,
        config.USE_GPU,
        config.MODEL_PATH,
        config.SCALER_PATH,
        'data/processed/loss_curve.png',
        seed=42,
        use_huber_loss=config.USE_HUBER_LOSS,
        num_heads=config.NUM_HEADS,
        huber_delta=config.HUBER_DELTA
    )
    print(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    main()
