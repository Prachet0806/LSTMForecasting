# config.py
# Configuration file for LSTM Stock Price Prediction (Unidirectional for Real-World Forecasting)
from data_utils import get_num_features

# Model Architecture Parameters
SEQ_LENGTH = 30
INPUT_SIZE = get_num_features()  # Dynamically set based on processed data (12 features: Close, Open, High, Low, Volume, MACD, RSI, EMA_12, EMA_26, BB_upper, BB_lower, BB_position)
HIDDEN_SIZE = 256
NUM_LAYERS = 4
OUTPUT_SIZE = 1
NUM_HEADS = 8  # For multi-head attention

# Training Parameters
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 100
DROPOUT = 0.1
USE_HUBER_LOSS = True  # Use Huber loss instead of MSE for robustness
HUBER_DELTA = 1.0  # Delta parameter for Huber loss

# Hardware Configuration
USE_GPU = True  # Set to True to use GPU (e.g., RTX4060)
CUDA_VISIBLE_DEVICES = '0'  # Set to GPU index if multiple GPUs (e.g., '0' for first GPU)

# File Paths
MODEL_PATH = "data/processed/lstm_model.pth"
SCALER_PATH = "data/processed/y_scaler.pkl"
DATA_PATH = "data/raw/AAPL_stock_data_clean.csv"

# Hyperparameter search space (for random search - hyperparam_search.py)
HYPERPARAM_SEARCH = {
    'SEQ_LENGTH': [30, 60, 100, 120],
    'HIDDEN_SIZE': [64, 128, 256],
    'NUM_LAYERS': [2, 3, 4],
    'BATCH_SIZE': [16, 32, 64],
    'LR': [0.001, 0.0005, 0.0001],
    'DROPOUT': [0.1, 0.2, 0.3],
    'NUM_HEADS': [4, 8, 16],
    'USE_HUBER_LOSS': [False, True],
    'HUBER_DELTA': [0.5, 1.0, 1.5, 2.0],
}

# Optuna search space (for Bayesian optimization - optuna_search.py)
OPTUNA_SEARCH_SPACE = {
    'SEQ_LENGTH': [30, 60, 100, 120],
    'HIDDEN_SIZE': [64, 128, 256, 512],
    'NUM_LAYERS': [2, 3, 4, 5],
    'BATCH_SIZE': [16, 32, 64, 128],
    'LR_RANGE': [1e-4, 1e-2],  # Log scale range for Optuna
    'DROPOUT_RANGE': [0.1, 0.5],  # Continuous range for Optuna
    'NUM_HEADS': [4, 8, 16, 32],
    'USE_HUBER_LOSS': [False, True],
    'HUBER_DELTA_RANGE': [0.1, 3.0],  # Continuous range for Optuna
}

# Optimization Configuration
N_SEARCH = 12  # Number of trials for random search
N_OPTUNA_TRIALS = 20  # Number of trials for Optuna Bayesian optimization

# Early Stopping Configuration
EARLY_STOPPING_PATIENCE = 15
LEARNING_RATE_PATIENCE = 5
LEARNING_RATE_FACTOR = 0.5

# Data Configuration
TRAIN_VAL_SPLIT = 0.9  # Use 90% for training, 10% for validation (chronological)
TEST_SPLIT = 0.8  # Use 80% for training+validation, 20% for testing
