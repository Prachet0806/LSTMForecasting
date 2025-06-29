# optuna_analysis.py
import optuna
import matplotlib.pyplot as plt
import pickle
import os

def analyze_study(study_path='data/processed/optuna_study.pkl'):
    """Analyze Optuna study results"""
    
    if not os.path.exists(study_path):
        print(f"Study file not found: {study_path}")
        return
    
    # Load study
    with open(study_path, 'rb') as f:
        study = pickle.load(f)
    
    print("=== Optuna Study Analysis ===")
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_trial.value:.6f}")
    print(f"Best params: {study.best_trial.params}")
    
    # Get trial states
    completed_trials = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    pruned_trials = study.get_trials(states=[optuna.trial.TrialState.PRUNED])
    failed_trials = study.get_trials(states=[optuna.trial.TrialState.FAIL])
    
    print(f"\nTrial Statistics:")
    print(f"Completed: {len(completed_trials)}")
    print(f"Pruned: {len(pruned_trials)}")
    print(f"Failed: {len(failed_trials)}")
    
    # Create analysis plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Optimization history
    optuna.visualization.matplotlib.plot_optimization_history(study, ax=axes[0, 0])
    axes[0, 0].set_title("Optimization History")
    
    # Plot 2: Parameter importance
    try:
        optuna.visualization.matplotlib.plot_param_importances(study, ax=axes[0, 1])
        axes[0, 1].set_title("Parameter Importance")
    except:
        axes[0, 1].text(0.5, 0.5, "Parameter importance\nnot available", 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title("Parameter Importance")
    
    # Plot 3: Parallel coordinate plot
    try:
        optuna.visualization.matplotlib.plot_parallel_coordinate(study, ax=axes[1, 0])
        axes[1, 0].set_title("Parallel Coordinate Plot")
    except:
        axes[1, 0].text(0.5, 0.5, "Parallel coordinate plot\nnot available", 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title("Parallel Coordinate Plot")
    
    # Plot 4: Contour plot (if 2+ parameters)
    try:
        if len(study.best_trial.params) >= 2:
            param_names = list(study.best_trial.params.keys())[:2]
            optuna.visualization.matplotlib.plot_contour(study, params=param_names, ax=axes[1, 1])
            axes[1, 1].set_title(f"Contour Plot: {param_names[0]} vs {param_names[1]}")
        else:
            axes[1, 1].text(0.5, 0.5, "Contour plot requires\n2+ parameters", 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title("Contour Plot")
    except:
        axes[1, 1].text(0.5, 0.5, "Contour plot\nnot available", 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Contour Plot")
    
    plt.tight_layout()
    plt.savefig('data/processed/optuna_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print top trials
    print(f"\n=== Top 5 Trials ===")
    trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
    for i, trial in enumerate(trials[:5]):
        print(f"{i+1}. Trial {trial.number}: {trial.value:.6f} | {trial.params}")

if __name__ == "__main__":
    analyze_study() 
