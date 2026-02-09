import optuna
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from models import ModelWrapper
from helper_functions import (_get_model_instance, balanced_class_weights, check_study_existence, 
                              find_optimal_batch_size, run_trial_with_cleanup, calculate_metrics, 
                              calculate_pr_metrics_batched, save_pr_artifacts, save_metrics_to_pickle, 
                              save_dataframe_to_pickle)
from utilities import FocalLoss, set_seed
from training_functions import train_and_validate_with_loader, train_and_validate
#from training_funcs import train_and_validate
import pandas as pd
import os
from datetime import datetime
from torch_geometric.loader import NeighborLoader

DEFAULT_EARLY_STOP = {
    "patience": 20,
    "min_delta": 1e-3,
}
EARLY_STOP_LOGGING = False

def _early_stop_args_from(source: dict) -> dict:
    """Build early stopping kwargs, falling back to defaults when keys are absent."""
    return {
        "patience": source.get("early_stop_patience", DEFAULT_EARLY_STOP["patience"]),
        "min_delta": source.get("early_stop_min_delta", DEFAULT_EARLY_STOP["min_delta"]),
        "log_early_stop": EARLY_STOP_LOGGING,
    }

def hyperparameter_tuning(
        models: list[str],
        dataset_name: str,
        data,
        masks: dict[str, torch.Tensor]
        ):
    print(f"Starting hyperparameter tuning for dataset: {dataset_name}")
    model_parameters = {model_name: [] for model_name in models}
    METRIC_KEYS = [
        'accuracy', 'precision', 'precision_illicit', 'recall', 'recall_illicit',
        'f1', 'f1_illicit', 'roc_auc', 'PRAUC', 'kappa'
    ]
    wrapper_models = ['MLP', 'GCN', 'GAT', 'GIN']
    sklearn_models = ['SVM', 'XGB', 'RF']

    for model_name in tqdm(models, desc="Models", unit="model"):
        if model_name in wrapper_models:
            n_trials = 100
        else:
            n_trials = 50
        study_name = f'{model_name}_optimization on {dataset_name} dataset'
        db_path = f'sqlite:///optimization_results_on_{dataset_name}.db'
        if check_study_existence(model_name, dataset_name):
            print(f"Study for {model_name} on {dataset_name} already exists. Skipping optimization.")
            continue
        else:
            print(f"Starting optimization for {model_name} on {dataset_name} with {n_trials} trials.")
            try:
                if dataset_name in ["Elliptic", "AMLSim", "IBM_AML_HiSmall", "IBM_AML_LiSmall"]:
                    #Do not use neighbourloader for these datasets
                    if check_study_existence(model_name, dataset_name): 
                        study = optuna.load_study(study_name=study_name, storage=db_path)
                    else:
                        study = optuna.create_study(
                            direction='maximize',
                            study_name=study_name,
                            storage=db_path,
                            load_if_exists=True
                        )
                        with tqdm(total=n_trials, desc=f"{model_name} trials", leave=False, unit="trial") as trial_bar:
                            def _optuna_progress_callback(study_inner, trial):
                                trial_bar.update()
                            
                            alpha_focal = balanced_class_weights(data.y.cpu().numpy())
                            study.optimize(
                                lambda trial: run_trial_with_cleanup(
                                    objective, model_name, trial, model_name, data, alpha_focal=alpha_focal, dataset_name=dataset_name),
                                    n_trials=n_trials,
                                    callbacks=[_optuna_progress_callback]
                                )
                    model_parameters[model_name].append(study.best_params)
                    print(f"Best F1-illicit for {model_name} on {dataset_name}: {study.best_value:.4f}")

            except Exception as e:
                print(f"\n{'!'*80}")
                print(f"✗ ERROR processing dataset {dataset_name}:")
                print(f"  {type(e).__name__}: {str(e)}")
                print(f"  Continuing to next dataset...")
                print(f"{'!'*80}\n")
                
                # Clean up GPU memory even on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                continue

    return model_parameters

def objective(trial, model: str, data=None, alpha_focal=None, dataset_name=None, masks=None):
    # Get model instance with trial-suggested hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.05, log=False)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    gamma_focal = trial.suggest_float('gamma_focal', 0.1, 5.0)
    train_mask = masks['train_mask']
    # Use pre-calculated alpha_focal if provided, else calculate (fallback)

    if alpha_focal is None:
        device = data.y.device if data is not None else torch.device('cpu')
        alpha_focal = balanced_class_weights(data.y[train_mask]).to(device)
    
    # Validate alpha_focal has correct shape (should be [2] for binary classification)
    if alpha_focal.shape[0] != 2:
        raise ValueError(f"alpha_focal should have 2 elements for binary classification, got {alpha_focal.shape[0]}")
        
    criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal, reduction='mean')
    
    early_stop_patience = trial.suggest_int('early_stop_patience', 5, 40)
    early_stop_min_delta = trial.suggest_float('early_stop_min_delta', 1e-4, 5e-3, log=True)
    trial_early_stop_args = _early_stop_args_from({
        "early_stop_patience": early_stop_patience,
        "early_stop_min_delta": early_stop_min_delta
    })
    
    # Pass train_mask to _get_model_instance for XGB leakage prevention
    model_instance = _get_model_instance(trial, model, data, device, train_mask=train_mask)

    wrapper_models = ['MLP', 'GCN', 'GAT', 'GIN']
    sklearn_models = ['SVM', 'XGB', 'RF']
    
    if model in sklearn_models:
        num_epochs = 50  
    elif model == "MLP":
        num_epochs = 50
    else: # GNNs
        num_epochs = 400
    
    
    if model in wrapper_models:
        optimiser = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_wrapper = ModelWrapper(model_instance, optimiser, criterion)
        model_wrapper.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        if dataset_name in ["Elliptic", "AMLSim", "IBM_AML_HiSmall", "IBM_AML_LiSmall"]:
            #Do not use neighbourloader for these datasets
            #I am here and struggling with the model wrapper, train_and_validate and neighbourloader interactions.