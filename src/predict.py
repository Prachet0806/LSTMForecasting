# predict.py
# This script loads a pre-trained LSTM model and uses it to make predictions on test data.
# It saves the inverse-transformed predictions to a file for later evaluation.

import torch
import numpy as np
import joblib
from model import LSTMModel
from data_utils import load_processed_data
import config
import json

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, X_test, _, _, _, y_scaler = load_processed_data()
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Load best hyperparameters
    with open("data/processed/best_hparams.json", "r") as f:
        best_hparams = json.load(f)

    model = LSTMModel(
        input_size=X_test.shape[2],
        hidden_size=best_hparams['HIDDEN_SIZE'],
        num_layers=best_hparams['NUM_LAYERS'],
        dropout=best_hparams['DROPOUT'],
        output_size=1,
        use_attention=True,
        num_heads=best_hparams.get('NUM_HEADS', config.NUM_HEADS)  # Use config default if not in best_hparams
    )
    model.load_state_dict(torch.load("data/processed/best_lstm_model.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    with torch.no_grad():
        predictions = model(X_tensor).cpu().numpy()

    # Inverse-transform predictions using y_scaler
    predictions_inv = y_scaler.inverse_transform(predictions)
    np.save("data/processed/y_pred.npy", predictions_inv)
    print("Predictions saved to data/processed/y_pred.npy.")

if __name__ == "__main__":
    main()
