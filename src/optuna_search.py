# optuna_search.py
import optuna
import config
import os
import json
import torch
from train import train
from data_prep import prepare_data
import numpy as np

def objective(trial):
    """Objective function for Optuna optimization"""
    
    # Define hyperparameter search space
    params = {
        'SEQ_LENGTH': trial.suggest_categorical('SEQ_LENGTH', [30, 60, 100, 120]),
        'HIDDEN_SIZE': trial.suggest_categorical('HIDDEN_SIZE', [64, 128, 256, 512]),
        'NUM_LAYERS': trial.suggest_int('NUM_LAYERS', 2, 5),
        'BATCH_SIZE': trial.suggest_categorical('BATCH_SIZE', [16, 32, 64, 128]),
        'LR': trial.suggest_float('LR', 1e-4, 1e-2, log=True),
        'DROPOUT': trial.suggest_float('DROPOUT', 0.1, 0.5),
        'NUM_HEADS': trial.suggest_categorical('NUM_HEADS', [4, 8, 16, 32]),
        'USE_HUBER_LOSS': trial.suggest_categorical('USE_HUBER_LOSS', [False, True]),
        'HUBER_DELTA': trial.suggest_float('HUBER_DELTA', 0.1, 3.0),
    }
    
    print(f"\nTrial {trial.number}: {params}")
    
    try:
        # Regenerate data if sequence length changed (for first trial or when it changes)
        if not hasattr(objective, 'current_seq_length') or params['SEQ_LENGTH'] != objective.current_seq_length:
            print(f"Regenerating data for sequence length {params['SEQ_LENGTH']}...")
            prepare_data(seq_length=params['SEQ_LENGTH'])
            objective.current_seq_length = params['SEQ_LENGTH']
        
        # Train model with current hyperparameters
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
            model_path='data/processed/temp_model.pth',
            scaler_path=None,
            plot_loss_path=None,
            seed=42,
            use_huber_loss=params['USE_HUBER_LOSS'],
            num_heads=params['NUM_HEADS'],
            huber_delta=params['HUBER_DELTA']
        )
        
        # Report intermediate value for pruning
        trial.report(val_loss, step=1)
        
        # Prune if the trial is not promising
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return val_loss
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.TrialPruned()

def main():
    # Set up Optuna study
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        ),
        storage=None  # In-memory storage
    )
    
    # Run optimization
    print("Starting Optuna optimization...")
    print(f"Using GPU: {torch.cuda.is_available()} (Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    study.optimize(objective, n_trials=config.N_OPTUNA_TRIALS, timeout=None)
    
    # Get best results
    best_trial = study.best_trial
    best_params = best_trial.params
    best_loss = best_trial.value
    
    # Check if we have a valid best loss
    if best_loss is None:
        print("No valid trials completed. Exiting.")
        return
    
    print("\n=== Optuna Optimization Complete ===")
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Best hyperparameters: {best_params}")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))}")
    
    # Check if this is better than previous best
    best_hparams_path = 'data/processed/best_hparams.json'
    best_loss_path = 'data/processed/best_val_loss.txt'
    best_model_path = 'data/processed/best_lstm_model.pth'
    
    # Load previous best loss if exists
    previous_best_loss = float('inf')
    if os.path.exists(best_loss_path):
        with open(best_loss_path, 'r') as f:
            try:
                previous_best_loss = float(f.read().strip())
                print(f"Previous best validation loss: {previous_best_loss:.6f}")
            except Exception:
                previous_best_loss = float('inf')
    
    # Only save if this run found a better model
    if best_loss < previous_best_loss:
        print(f"New best model found! Previous: {previous_best_loss:.6f}, New: {best_loss:.6f}")
        
        # Save best hyperparameters
        with open(best_hparams_path, 'w') as f:
            json.dump(best_params, f)
        with open(best_loss_path, 'w') as f:
            f.write(str(best_loss))
        
        # Train final model with best parameters
        print("\nTraining final model with best hyperparameters...")
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
    else:
        print(f"No improvement found. Keeping previous best model (loss: {previous_best_loss:.6f})")
    
    # Optional: Save study for later analysis
    study_path = 'data/processed/optuna_study.pkl'
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"Study saved to {study_path}")

if __name__ == "__main__":
    main() 
