import os
import secrets
import torch
import numpy as np
import pandas as pd
import optuna
# Move local imports to global scope if possible, or keep at top of file
# from utilities import load_batch_size_by_phase
# from sklearn.metrics import accuracy_score, f1_score
from helper_functions import *
from helper_functions import _get_model_instance
from training_functions import train_and_validate
from utilities import *
from models import ModelWrapper

def run_final_evaluation(models, model_parameters, data, data_for_optimisation, masks):
    print(f"\nStarting FINAL EVALUATION for {data_for_optimisation} dataset...")
    
    # Optimisation: Hoisted directory creation outside the loop
    os.makedirs(f"results/{data_for_optimisation}/pr_curves", exist_ok=True)
    os.makedirs(f"results/{data_for_optimisation}/metrics", exist_ok=True)
    os.makedirs(f"results/{data_for_optimisation}/pkl_files", exist_ok=True)
        
    for model_name in models:
        if model_name not in model_parameters or not model_parameters[model_name]:
            print(f"No parameters found for {model_name}, skipping evaluation.")
            continue
            
        best_params = model_parameters[model_name][-1]
        print(f"Evaluating {model_name} on {data_for_optimisation}...")
        
        evaluate_model_performance(
            model_name=model_name,
            best_params=best_params,
            data=data,
            masks=masks,
            dataset_name=data_for_optimisation,
            n_runs=30
        )

def evaluate_model_performance(model_name, best_params, data, masks, dataset_name, n_runs=30):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapper_models = {'MLP', 'GCN', 'GAT', 'GIN'} # Optimisation: Set lookup is O(1)
    sklearn_models = {'SVM', 'RF'}
    gpu_sklearn_models = {'XGB'}
    
    # Extract regular masks from masks variable
    if data.y.device != masks.device:
        print("Masks on differing device than data.y. Check data input to evaluate_model_performance.")
        raise ValueError("Masks and data.y must be on the same device.")


    # Store regular masks in easily accessible format
    regular_masks = [masks["train_mask"], masks["val_mask"], masks["test_mask"]]

    # Optimisation: Lazy-load CPU arrays ONLY if using an sklearn model
    sklearn_data = None

    if dataset_name == "Elliptic":
        if model_name in sklearn_models:
            # Extract performance evaluation masks for Elliptic (excludes nodes with y=-1)
            train_perf_eval_mask = masks["train_perf_eval_mask"]
            val_perf_eval_mask = masks["val_perf_eval_mask"]
            test_perf_eval_mask = masks["test_perf_eval_mask"]

            # Store perf_eval masks in easily accessible format
            perf_eval_masks = [train_perf_eval_mask, val_perf_eval_mask, test_perf_eval_mask]

            sklearn_data = {
                'train_x': data.x[regular_masks[0]].cpu().numpy(),
                'train_perf_x': data.x[perf_eval_masks[0]].cpu().numpy(),
                'train_y': data.y[regular_masks[0]].cpu().numpy(),
                'train_perf_y': data.y[perf_eval_masks[0]].cpu().numpy(),
                'val_x': data.x[regular_masks[1]].cpu().numpy(),
                'val_perf_x': data.x[perf_eval_masks[1]].cpu().numpy(),
                'val_y': data.y[regular_masks[1]].cpu().numpy(),
                'val_perf_y': data.y[perf_eval_masks[1]].cpu().numpy(),
                'test_x': data.x[regular_masks[2]].cpu().numpy(),
                'test_perf_x': data.x[perf_eval_masks[2]].cpu().numpy(),
                'test_y': data.y[regular_masks[2]].cpu().numpy(),
                'test_perf_y': data.y[perf_eval_masks[2]].cpu().numpy()
            }
        elif model_name in gpu_sklearn_models:
            # Extract performance evaluation masks for Elliptic (excludes nodes with y=-1)
            train_perf_eval_mask = masks["train_perf_eval_mask"]
            val_perf_eval_mask = masks["val_perf_eval_mask"]
            test_perf_eval_mask = masks["test_perf_eval_mask"]

            # Store perf_eval masks in easily accessible format
            perf_eval_masks = [train_perf_eval_mask, val_perf_eval_mask, test_perf_eval_mask]

            sklearn_data = {
                'train_x': data.x[regular_masks[0]],
                'train_perf_x': data.x[perf_eval_masks[0]],
                'train_y': data.y[regular_masks[0]],
                'train_perf_y': data.y[perf_eval_masks[0]],
                'val_x': data.x[regular_masks[1]],
                'val_perf_x': data.x[perf_eval_masks[1]],
                'val_y': data.y[regular_masks[1]],
                'val_perf_y': data.y[perf_eval_masks[1]],
                'test_x': data.x[regular_masks[2]],
                'test_perf_x': data.x[perf_eval_masks[2]],
                'test_y': data.y[regular_masks[2]],
                'test_perf_y': data.y[perf_eval_masks[2]]
            }
    else:
        if model_name in sklearn_models:
            sklearn_data = {
                'train_x': data.x[regular_masks[0]].cpu().numpy(),
                'train_y': data.y[regular_masks[0]].cpu().numpy(),
                'val_x': data.x[regular_masks[1]].cpu().numpy(),
                'val_y': data.y[regular_masks[1]].cpu().numpy(),
                'test_x': data.x[regular_masks[2]].cpu().numpy(),
                'test_y': data.y[regular_masks[2]].cpu().numpy()
            }
        elif model_name in gpu_sklearn_models:
            sklearn_data = {
                'train_x': data.x[regular_masks[0]],
                'train_y': data.y[regular_masks[0]],
                'val_x': data.x[regular_masks[1]],
                'val_y': data.y[regular_masks[1]],
                'test_x': data.x[regular_masks[2]],
                'test_y': data.y[regular_masks[2]]
            }
    
    # Optimisation: Pre-calculate batch sizes and NeighborLoaders OUTSIDE the seed loop
    train_loader, val_loader, test_loader = None, None, None
    alpha_focal = None
    batch_loader_datasets = ["IBM_AML_HiMedium", "IBM_AML_LiMedium"]
    detailed_metrics = []
    if model_name in wrapper_models:
        if dataset_name not in batch_loader_datasets:
            print(f"  > Using FULL BATCH for {model_name} on {dataset_name} (no NeighborLoader)")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            alpha_focal = balanced_class_weights(data.y[regular_masks[0]]).to(device)
            n_runs = 10 if model_name == 'RF' else 30
            for i in range(n_runs):
                run_id = f"run_{i+1}"
                print(f"  > Run {i+1}/{n_runs}")
                
                #Running fixed trial as I already have the correct hyperparameters
                fixed_trial = optuna.trial.FixedTrial(best_params)
                model_instance = _get_model_instance(fixed_trial, model_name, data, device, train_mask = masks['train_mask'])
                y_true = data.y[regular_masks[2] if dataset_name != "Elliptic" else masks["train_perf_eval_mask"]]
                y_pred, y_probs = None, None
                run_metrics = {}
                learning_rate = best_params.get('learning_rate', 0.01)
                weight_decay = best_params.get('weight_decay', 5e-4)
                gamma_focal = best_params.get('gamma_focal', 2.0)
                patience = best_params.get('early_stop_patience', 20)
                min_delta = best_params.get('early_stop_min_delta', 1e-3)
                
                criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal, reduction='mean')
                optimiser = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
                
                model_wrapper = ModelWrapper(model_instance, optimiser, criterion)
                model_wrapper.model.to(device)
                
                num_epochs = 50 if model_name == "MLP" else 150
                
                best_f1_model_wts, _ = train_and_validate(
                                        model_wrapper,
                                        data,
                                        masks,   
                                        num_epochs,
                                        dataset_name,
                                        patience=patience,
                                        min_delta=min_delta,
                                        log_early_stop=False
                                    )
                model_wrapper.model.load_state_dict(best_wts)
                
                if dataset_name == "Elliptic":
                    masks = [masks['test_mask'], masks['test_perf_eval_mask']]
                    test_loss, test_metrics = model_wrapper.evaluate_elliptic(data, )
                    
                
                y_probs = test_metrics['probs'] 
                y_pred = test_metrics['preds']
                y_true = test_metrics['y_true'] 
                run_metrics = test_metrics 
                
        else:
        
            from utilities import load_batch_size_by_phase
            alpha_focal = balanced_class_weights(data.y[regular_masks[0]]).to(device)
            
            print(f"Finding optimal EVALUATION batch size for {model_name}...")
            eval_batch_size = load_batch_size_by_phase(dataset_name, model_name, phase='evaluation')
            
            if eval_batch_size is None:
                def model_builder_for_eval_size_check():
                    class Mocktrial:
                        def suggest_int(self, name, low, high, step=None): return high
                        def suggest_float(self, name, low, high, log=False): return low
                        def suggest_categorical(self, name, choices): return choices[0]
                    return _get_model_instance(Mocktrial(), model_name, data, device, train_mask=train_mask)
                
                eval_batch_size = find_optimal_batch_size(
                    model_builder_for_eval_size_check, data, device, train_mask,
                    num_neighbors=[10, 10], dataset_name=dataset_name, model_name=model_name, phase='evaluation'
                )
            print(f"  > Using EVALUATION batch size {eval_batch_size} for {model_name}")

            tuning_batch_size = load_batch_size_by_phase(dataset_name, model_name, phase='tuning') or 65536
            if tuning_batch_size == 65536:
                print(f"  > Using default TUNING batch size 65536 (no cached value found)")
                
            num_neighbors = [10, 10]
            train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=tuning_batch_size, input_nodes=train_mask)
            val_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=eval_batch_size, input_nodes=val_mask)
            test_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=eval_batch_size, input_nodes=test_mask)

    detailed_metrics = []
    n_runs = 30 if model_name == 'RF' else 30
    seeds = [secrets.randbits(32) for _ in range(n_runs)] 
    
    for i, seed in enumerate(seeds):
        set_seed(seed)
        print(f"  > Run {i+1}/{n_runs} (Seed {seed})")
    
        fixed_trial = optuna.trial.FixedTrial(best_params)
        model_instance = _get_model_instance(fixed_trial, model_name, data, device, train_mask=train_mask)
        
        y_true = data.y[test_mask].cpu().numpy()
        y_pred, y_probs = None, None
        run_metrics = {}
        
        if model_name in wrapper_models:
            learning_rate = best_params.get('learning_rate', 0.01)
            weight_decay = best_params.get('weight_decay', 5e-4)
            gamma_focal = best_params.get('gamma_focal', 2.0)
            patience = best_params.get('early_stop_patience', 20)
            min_delta = best_params.get('early_stop_min_delta', 1e-3)
            
            criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal, reduction='mean')
            optimiser = torch.optim.Adam(model_instance.parameters(), lr=learning_rate, weight_decay=weight_decay)
            
            model_wrapper = ModelWrapper(model_instance, optimiser, criterion)
            model_wrapper.model.to(device)
            
            num_epochs = 50 if model_name == "MLP" else 150
            
            best_wts, _ = train_and_validate(model_wrapper, train_loader, val_loader, num_epochs, patience=patience, min_delta=min_delta, log_early_stop=False)
            model_wrapper.model.load_state_dict(best_wts)
            
            _, test_metrics = model_wrapper.evaluate(test_loader)
            
            y_probs = test_metrics['probs'] 
            y_pred = test_metrics['preds']
            y_true = test_metrics['y_true'] 
            run_metrics = test_metrics 
            
        elif model_name in sklearn_models:
            model_instance.fit(sklearn_data['train_x'], sklearn_data['train_y'])
            
            try:
                y_probs = model_instance.predict_proba(sklearn_data['test_x'])
            except AttributeError:
                # Optimisation: Fallback to scaled decision_function for SVM PR curves
                if hasattr(model_instance, 'decision_function'):
                    dfunc = model_instance.decision_function(sklearn_data['test_x'])
                    y_probs = (dfunc - dfunc.min()) / (dfunc.max() - dfunc.min())
                else:
                    y_probs = None
            
            y_pred = model_instance.predict(sklearn_data['test_x'])
            
        if model_name not in wrapper_models:
            if y_probs is not None:
                run_metrics = calculate_metrics(y_true, y_pred, y_probs)
            else:
                from sklearn.metrics import accuracy_score, f1_score
                run_metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': np.nan,
                    'precision_illicit': np.nan,
                    'recall': np.nan,
                    'recall_illicit': np.nan,
                    'f1': f1_score(y_true, y_pred, average='weighted'),
                    'f1_illicit': f1_score(y_true, y_pred, pos_label=1, average='binary'),
                    'roc_auc': np.nan,
                    'PRAUC': np.nan,
                    'kappa': np.nan,
                }
        
        if y_probs is not None:
            # Optimisation: Handle dimension checking dynamically
            y_probs_for_pr = y_probs[:, 1] if getattr(y_probs, 'ndim', 1) == 2 else y_probs
            y_probs_tensor = torch.as_tensor(y_probs_for_pr, dtype=torch.float32)
            y_true_tensor = torch.as_tensor(y_true, dtype=torch.long)
            
            precision, recall, thresholds = calculate_pr_metrics_batched(y_probs_tensor, y_true_tensor)
            
            pr_filename = f"results/{dataset_name}/pr_curves/{model_name}_run_{i+1}"
            pr_auc = save_pr_artifacts(precision, recall, thresholds, pr_filename)
            run_metrics['PRAUC'] = pr_auc
            
            pr_data = {
                'precision': precision.numpy(),
                'recall': recall.numpy(),
                'thresholds': thresholds.numpy(),
                'auc': pr_auc,
                'y_true': y_true,
                'y_probs': y_probs_for_pr
            }
            save_metrics_to_pickle(pr_data, f"results/{dataset_name}/pkl_files/{model_name}_run_{i+1}_pr_data.pkl")
        
        run_metrics['model'] = model_name
        run_metrics['run'] = i + 1
        detailed_metrics.append(run_metrics)
        
        save_metrics_to_pickle(run_metrics, f"results/{dataset_name}/pkl_files/{model_name}_run_{i+1}_metrics.pkl")
        
    df = pd.DataFrame(detailed_metrics)
    
    df.to_csv(f"results/{dataset_name}/metrics/{model_name}_detailed_metrics.csv", index=True)
    save_dataframe_to_pickle(df, f"results/{dataset_name}/pkl_files/{model_name}_detailed_metrics.pkl")
    
    columns_to_drop = {'model', 'run', 'probs', 'preds', 'y_true'}
    numeric_df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')
    summary = numeric_df.agg(['mean', 'std']).transpose()
    summary['model'] = model_name
    
    save_dataframe_to_pickle(summary, f"results/{dataset_name}/pkl_files/{model_name}_summary_metrics.pkl")
    
    summary_file = f"results/{dataset_name}/metrics/summary_metrics.csv"
    summary.to_csv(summary_file, mode='a' if os.path.exists(summary_file) else 'w', header=not os.path.exists(summary_file))
        
    print(f"  > Completed. Metrics saved to results/{dataset_name}/metrics/")