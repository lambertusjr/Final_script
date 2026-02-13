import optuna
import traceback
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
        models,
        dataset_name,
        data,
        masks
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
                            
                            # Bug 19 fix: balanced_class_weights expects a tensor, not numpy array
                            alpha_focal = balanced_class_weights(data.y)
                            # Bug 13 fix: masks was missing from lambda — objective() requires it
                            study.optimize(
                                lambda trial: run_trial_with_cleanup(
                                    objective, model_name, trial, model_name, data, alpha_focal=alpha_focal, dataset_name=dataset_name, masks=masks),
                                    n_trials=n_trials,
                                    callbacks=[_optuna_progress_callback]
                                )
                    model_parameters[model_name].append(study.best_params)
                    print(f"Best F1-illicit for {model_name} on {dataset_name}: {study.best_value:.4f}")

            except Exception as e:
                print(f"\n{'!'*80}")
                print(f"✗ ERROR processing model {model_name} on {dataset_name}:")
                print(f"  {type(e).__name__}: {str(e)}")
                traceback.print_exc()
                print(f"  Continuing to next model...")
                print(f"{'!'*80}\n")
                
                # Clean up GPU memory even on error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                continue

    return model_parameters

def objective(trial, model, data, alpha_focal, dataset_name, masks):
    # Get model instance with trial-suggested hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 0.005, 0.05, log=False)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    gamma_focal = trial.suggest_float('gamma_focal', 0.1, 5.0)
    train_mask = masks['train_mask']
    # Bug 15 fix: device must be defined before any conditional block that uses it
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use pre-calculated alpha_focal if provided, else calculate (fallback)
    if alpha_focal is None:
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
        
    # Bug 14 fix: train_and_validate expects a dict (calls .items()), not a list
    if dataset_name == "Elliptic":
        train_val_masks = {
            'train_mask': masks['train_mask'],
            'train_perf_eval_mask': masks['train_perf_eval_mask'],
            'val_mask': masks['val_mask'],
            'val_perf_eval_mask': masks['val_perf_eval_mask'],
        }
    else:
        train_val_masks = {
            'train_mask': masks['train_mask'],
            'val_mask': masks['val_mask'],
        }
    
    
    if model in wrapper_models:
        optimiser = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_wrapper = ModelWrapper(model_instance, optimiser, criterion)
        model_wrapper.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        if dataset_name in ["Elliptic", "AMLSim", "IBM_AML_HiSmall", "IBM_AML_LiSmall"]:
            # Bug 14 cont: pass train_val_masks dict instead of list
            best_f1_model_wts, best_f1 = train_and_validate(
                model_wrapper, data, train_val_masks, num_epochs, dataset_name, **trial_early_stop_args
            )
        else:
            # Bug 17 fix: fail loudly instead of silently returning None
            raise NotImplementedError("NeighborLoader tuning not yet implemented")
        # Bug 18 fix: return best_f1 so Optuna can use it
        return best_f1

    elif model in sklearn_models:
        gpu_enabled_models = ['XGB']
        if model in gpu_enabled_models:
            train_x = data.x[masks['train_mask']]
            train_y = data.y[masks['train_mask']] 
            val_x = data.x[masks['val_mask']]
            val_y = data.y[masks['val_mask']]
        else:
            train_x = data.x[masks['train_mask']].cpu().numpy()
            train_y = data.y[masks['train_mask']].cpu().numpy() 
            val_x = data.x[masks['val_mask']].cpu().numpy()
            val_y = data.y[masks['val_mask']].cpu().numpy()
        
        model_instance.fit(train_x, train_y)
        pred = model_instance.predict(val_x)
        # Bug 16 fix: BinaryF1Score crashes on numpy arrays (no .device); use sklearn f1_score instead
        f1_illicit = f1_score(val_y, pred, pos_label=1, average='binary')
        return float(f1_illicit)

    else:
        raise ValueError(f"Unknown model type: {model}")