# evaluate.py
# This script evaluates the performance of the LSTM model on the test set.
import numpy as np
import matplotlib.pyplot as plt
import joblib
from data_utils import load_processed_data
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def main():
    y_pred = np.load("data/processed/y_pred.npy")
    _, _, _, y_test, _, y_scaler = load_processed_data()

    # (Optional) Load best hyperparameters for reference
    try:
        with open("data/processed/best_hparams.json", "r") as f:
            best_hparams = json.load(f)
        print(f"Evaluating with best hyperparameters: {best_hparams}")
    except Exception:
        pass

    # Inverse transform
    y_actual = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    y_pred_inv = y_pred.flatten()

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_inv))
    mae = mean_absolute_error(y_actual, y_pred_inv)
    mape = calculate_mape(y_actual, y_pred_inv)
    r2 = r2_score(y_actual, y_pred_inv)
    
    # Print comprehensive metrics
    print("\n=== Model Performance Metrics ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    
    # Residual analysis
    residuals = y_actual - y_pred_inv
    residual_mean = np.mean(residuals)
    residual_std = np.std(residuals)
    print(f"\nResidual Analysis:")
    print(f"Mean Residual: {residual_mean:.4f}")
    print(f"Std Residual: {residual_std:.4f}")
    print(f"Bias: {residual_mean:.4f}")

    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Predictions vs Actual
    axes[0, 0].plot(y_actual, label='Actual', alpha=0.7)
    axes[0, 0].plot(y_pred_inv, label='Predicted', alpha=0.7)
    axes[0, 0].set_title("LSTM Stock Price Forecast vs Actual (Test Set)")
    axes[0, 0].set_xlabel("Time Steps")
    axes[0, 0].set_ylabel("Price ($)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals over time
    axes[0, 1].plot(residuals, color='red', alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[0, 1].set_title("Residuals Over Time")
    axes[0, 1].set_xlabel("Time Steps")
    axes[0, 1].set_ylabel("Residuals ($)")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residuals histogram
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(x=residual_mean, color='red', linestyle='--', label=f'Mean: {residual_mean:.2f}')
    axes[1, 0].set_title("Residuals Distribution")
    axes[1, 0].set_xlabel("Residuals ($)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Actual vs Predicted scatter
    axes[1, 1].scatter(y_actual, y_pred_inv, alpha=0.6, color='green')
    min_val = min(y_actual.min(), y_pred_inv.min())
    max_val = max(y_actual.max(), y_pred_inv.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    axes[1, 1].set_title("Actual vs Predicted")
    axes[1, 1].set_xlabel("Actual Price ($)")
    axes[1, 1].set_ylabel("Predicted Price ($)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/processed/comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional residual analysis
    print(f"\n=== Residual Analysis Details ===")
    print(f"Positive residuals: {np.sum(residuals > 0)} ({np.sum(residuals > 0)/len(residuals)*100:.1f}%)")
    print(f"Negative residuals: {np.sum(residuals < 0)} ({np.sum(residuals < 0)/len(residuals)*100:.1f}%)")
    print(f"Max positive residual: {np.max(residuals):.4f}")
    print(f"Max negative residual: {np.min(residuals):.4f}")
    
    # Check for autocorrelation in residuals (simple lag-1 correlation)
    if len(residuals) > 1:
        lag1_corr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        print(f"Lag-1 autocorrelation: {lag1_corr:.4f}")
        if abs(lag1_corr) > 0.3:
            print("Warning: High autocorrelation in residuals detected!")

if __name__ == "__main__":
    main()
