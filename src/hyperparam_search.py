#hyperparam_search.py
import config
import os
os.environ['CUDA_VISIBLE_DEVICES'] = config.CUDA_VISIBLE_DEVICES
import itertools
import random
import json
from train import train
import torch
from data_prep import prepare_data
from itertools import product

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(42)
    
    # Create processed directory if it doesn't exist
    os.makedirs('data/processed', exist_ok=True)
    
    # File paths for saving best results
    best_hparams_path = 'data/processed/best_hparams.json'
    best_loss_path = 'data/processed/best_val_loss.txt'
    best_model_path = 'data/processed/best_lstm_model.pth'
    
    # Load previous best loss if exists
    best_loss = float('inf')
    if os.path.exists(best_loss_path):
        with open(best_loss_path, 'r') as f:
            try:
                best_loss = float(f.read().strip())
                print(f"Previous best validation loss: {best_loss:.6f}")
            except Exception:
                best_loss = float('inf')
    
    # Generate all combinations
    keys = list(config.HYPERPARAM_SEARCH.keys())
    values = list(config.HYPERPARAM_SEARCH.values())
    all_combinations = list(itertools.product(*values))
    
    # Shuffle combinations for random search
    random.shuffle(all_combinations)
    
    results = []
    N_SEARCH = config.N_SEARCH
    
    print(f"Starting hyperparameter search with {N_SEARCH} trials...")
    print(f"Using GPU: {torch.cuda.is_available()} (Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    # Track current sequence length to avoid unnecessary data regeneration
    current_seq_length = None
    
    for i, combo in enumerate(all_combinations[:N_SEARCH]):
        params = dict(zip(keys, combo))
        print(f"\nTrial {i+1}/{N_SEARCH}: {params}")
        
        # Regenerate data if sequence length changed
        if params['SEQ_LENGTH'] != current_seq_length:
            print(f"Regenerating data for sequence length {params['SEQ_LENGTH']}...")
            prepare_data(seq_length=params['SEQ_LENGTH'])
            current_seq_length = params['SEQ_LENGTH']
        
        val_loss = train(
            input_size=config.INPUT_SIZE,
            hidden_size=params['HIDDEN_SIZE'],
            num_layers=params['NUM_LAYERS'],
            dropout=params['DROPOUT'],
            output_size=config.OUTPUT_SIZE,
            batch_size=params['BATCH_SIZE'],
            lr=params['LR'],
            epochs=config.EPOCHS,
            use_gpu=config.USE_GPU,
            model_path=best_model_path+'.tmp',
            scaler_path=config.SCALER_PATH,
            plot_loss_path=None,
            seed=42,
            use_huber_loss=params['USE_HUBER_LOSS'],
            num_heads=params['NUM_HEADS'],
            huber_delta=params['HUBER_DELTA']
        )
        results.append((val_loss, params))
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = params
            # Overwrite best model, hparams, and loss
            os.replace(best_model_path+'.tmp', best_model_path)
            with open(best_hparams_path, 'w') as f:
                json.dump(params, f)
            with open(best_loss_path, 'w') as f:
                f.write(str(best_loss))
            print(f"New best loss: {best_loss:.6f} with params: {best_params}")
        else:
            # Remove temp file
            if os.path.exists(best_model_path+'.tmp'):
                os.remove(best_model_path+'.tmp')
    
    # Sort results by validation loss
    results.sort(key=lambda x: x[0])
    
    print("\n=== Hyperparameter Search Complete ===")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Best hyperparameters: {best_params}")
    print(f"\nTop 5 results:")
    for i, (loss, params) in enumerate(results[:5]):
        print(f"{i+1}. Loss: {loss:.6f}, Params: {params}")
    
    # Train final model with best parameters
    print(f"\nTraining final model with best hyperparameters...")
    final_val_loss = train(
        input_size=config.INPUT_SIZE,
        hidden_size=best_params['HIDDEN_SIZE'],
        num_layers=best_params['NUM_LAYERS'],
        dropout=best_params['DROPOUT'],
        output_size=config.OUTPUT_SIZE,
        batch_size=best_params['BATCH_SIZE'],
        lr=best_params['LR'],
        epochs=config.EPOCHS,
        use_gpu=config.USE_GPU,
        model_path=best_model_path,
        scaler_path=config.SCALER_PATH,
        plot_loss_path='data/processed/loss_curve.png',
        seed=42,
        use_huber_loss=best_params['USE_HUBER_LOSS'],
        num_heads=best_params['NUM_HEADS'],
        huber_delta=best_params['HUBER_DELTA']
    )
    
    print(f"Final model validation loss: {final_val_loss:.6f}")
    print("\nUpdated predict.py and evaluate.py to use the best model.")

if __name__ == "__main__":
    main() 
