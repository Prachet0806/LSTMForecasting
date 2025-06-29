# LSTM Stock Price Forecasting Project

A comprehensive deep learning project for predicting AAPL stock prices using advanced LSTM architectures with attention mechanisms, technical indicators, and hyperparameter optimization.

## 🚀 Features

### Core Architecture
- **Multi-layer LSTM** with configurable depth (2-5 layers)
- **Multi-head attention mechanism** for enhanced sequence modeling
- **Unidirectional LSTM** optimized for real-world forecasting scenarios
- **Huber loss** with tunable delta for robust training
- **Early stopping** and learning rate scheduling

### Technical Indicators
- **MACD** (Moving Average Convergence Divergence)
- **RSI** (Relative Strength Index)
- **EMA** (Exponential Moving Averages) - 12 and 26 periods
- **Bollinger Bands** (Upper, Lower, and Position)

### Advanced Features
- **GPU acceleration** support (CUDA)
- **Hyperparameter optimization** with two strategies:
  - Random search (`hyperparam_search.py`)
  - Bayesian optimization with Optuna (`optuna_search.py`)
- **Chronological validation** split for realistic evaluation
- **Comprehensive evaluation metrics**: RMSE, MAE, MAPE, R²
- **Residual analysis** with autocorrelation plots
- **Dynamic sequence length** tuning
- **Model persistence** with best performance tracking

## 📁 Project Structure

```
LSTMForecasting/
├── data/
│   ├── raw/
│   │   └── AAPL_stock_data.csv          # Raw stock data
│   └── processed/
│       ├── X_train.npy, X_test.npy      # Processed features
│       ├── y_train.npy, y_test.npy      # Processed targets
│       ├── X_scaler.pkl, y_scaler.pkl   # Scaler objects
│       ├── lstm_model.pth               # Trained model
│       ├── optuna_study.pkl             # Optuna study results
│       └── *.png                        # Evaluation plots
├── src/
│   ├── config.py                        # Configuration parameters
│   ├── data_prep.py                     # Data preprocessing pipeline
│   ├── data_utils.py                    # Data utility functions
│   ├── model.py                         # LSTM model definition
│   ├── train.py                         # Training script
│   ├── evaluate.py                      # Model evaluation
│   ├── predict.py                       # Prediction script
│   ├── hyperparam_search.py             # Random search optimization
│   ├── optuna_search.py                 # Bayesian optimization
│   └── optuna_analysis.py               # Optuna results analysis
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Prachet0806/LSTMForecasting/
   cd LSTMForecasting
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU support** (optional)
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 📊 Data Preparation

The project uses AAPL stock data with the following features:
- **Price data**: Open, High, Low, Close, Volume
- **Technical indicators**: MACD, RSI, EMA(12), EMA(26)
- **Bollinger Bands**: Upper, Lower, Position

### Data Processing Pipeline
```bash
python src/data_prep.py
```

This script:
- Loads raw AAPL data
- Calculates technical indicators
- Handles missing values
- Splits data chronologically (80% train/val, 20% test)
- Fits scalers on training data only
- Saves processed data and scalers

## 🏗️ Model Architecture

### LSTM Model
- **Input**: 12 features (price + technical indicators)
- **LSTM layers**: 2-5 layers with configurable hidden size (64-512)
- **Attention**: Multi-head attention with 4-32 heads
- **Output**: Single value prediction (next day's close price)

### Key Components
1. **Multi-layer LSTM**: Stacked LSTM layers with dropout
2. **Multi-head Attention**: Captures complex temporal dependencies
3. **Global Average Pooling**: Reduces sequence to fixed-size representation
4. **Fully Connected Layer**: Final prediction layer

## 🚀 Usage

### 1. Basic Training
```bash
python src/train.py
```

### 2. Hyperparameter Optimization

#### Random Search
```bash
python src/hyperparam_search.py
```

#### Bayesian Optimization (Optuna)
```bash
python src/optuna_search.py
```

### 3. Making Predictions
```bash
python src/predict.py
```

### 4. Model Evaluation
```bash
python src/evaluate.py
```

### 5. Analyze Optuna Results
```bash
python src/optuna_analysis.py
```

## ⚙️ Configuration

Key parameters in `src/config.py`:

### Model Architecture
- `SEQ_LENGTH`: Input sequence length (30-120)
- `HIDDEN_SIZE`: LSTM hidden size (64-512)
- `NUM_LAYERS`: Number of LSTM layers (2-5)
- `NUM_HEADS`: Attention heads (4-32)

### Training Parameters
- `BATCH_SIZE`: Training batch size (16-128)
- `LR`: Learning rate (0.0001-0.01)
- `EPOCHS`: Maximum training epochs (100)
- `DROPOUT`: Dropout rate (0.1-0.5)
- `HUBER_DELTA`: Huber loss delta (0.1-3.0)

### Hardware
- `USE_GPU`: Enable GPU acceleration
- `CUDA_VISIBLE_DEVICES`: GPU device selection

## 📈 Evaluation Metrics

The model provides comprehensive evaluation:

### Performance Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

### Analysis Plots
- **Loss curves**: Training and validation loss
- **Predictions vs Actual**: Time series comparison
- **Residual analysis**: Error distribution and autocorrelation
- **Comprehensive evaluation**: All metrics and plots combined

## 🔧 Advanced Features

### Hyperparameter Optimization
1. **Random Search**: Grid-based exploration
2. **Optuna Bayesian**: Intelligent parameter search with pruning
3. **Pruning**: Early termination of poor trials
4. **Best Model Preservation**: Maintains best performing model across runs

### Training Enhancements
- **Early stopping**: Prevents overfitting
- **Learning rate scheduling**: Adaptive learning rate
- **Chronological splits**: Realistic train/validation/test splits
- **Robust loss**: Huber loss for outlier resistance

### Model Persistence
- **Automatic saving**: Best model saved automatically
- **Hyperparameter tracking**: JSON storage of best parameters
- **Study persistence**: Optuna study saved for analysis

## 🎯 Performance

Best performance metrics:
- **RMSE**: ~5 (varies with hyperparameters)
- **MAE**: ~3.99
- **MAPE**: ~1.9%
- **R²**: ~0.95+

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce `BATCH_SIZE` or `HIDDEN_SIZE`
   - Set `USE_GPU = False` in config

2. **Model loading errors**
   - Ensure model architecture matches saved weights
   - Check hyperparameter consistency

3. **Poor performance**
   - Try different `SEQ_LENGTH` values
   - Adjust `HUBER_DELTA` for your data
   - Increase `NUM_LAYERS` or `HIDDEN_SIZE`

### Performance Tips
- Use GPU acceleration when available
- Start with Optuna search for best hyperparameters
- Monitor training curves for overfitting
- Consider ensemble methods for production use

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📚 References

- LSTM: Hochreiter & Schmidhuber (1997)
- Attention: Vaswani et al. (2017)
- Technical Indicators: Various financial analysis sources
- Optuna: Akiba et al. (2019)

---

**Note**: This model is designed for educational purposes. Always perform thorough backtesting and risk assessment before using any financial prediction model in real trading scenarios. 
