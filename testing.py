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
    # Store perf_eval masks for Elliptic in easily accessible format if dataset is Elliptic
    if dataset_name == "Elliptic":
        elliptic_perf_masks = [masks['train_perf_eval_mask'], masks['val_perf_eval_mask'], masks['test_perf_eval_mask']]
    # Optimisation: Lazy-load CPU arrays ONLY if using an sklearn model
    
    #Optimised cpu conversion to a simple function that also checks whether conversion is necessary based on model type.
    sklearn_data = None
    needs_conversion = model_name in sklearn_models
    convert = lambda t: t.cpu().numpy() if needs_conversion else t

    # Optimisation of data preparation for sklearn models.
    if model_name in sklearn_models or model_name in gpu_sklearn_models:
        splits = ['train', 'val', 'test']
        sklearn_data = {}
        if dataset_name == "Elliptic":
            for i, split in enumerate(splits):
                sklearn_data[f'{split}_x'] = convert(data.x[regular_masks[i]])
                sklearn_data[f'{split}_perf_x'] = convert(data.x[elliptic_perf_masks[i]])
                sklearn_data[f'{split}_y'] = convert(data.y[regular_masks[i]])
                sklearn_data[f'{split}_perf_y'] = convert(data.y[elliptic_perf_masks[i]])
        else:
            for i, split in enumerate(splits):
                sklearn_data[f'{split}_x'] = convert(data.x[regular_masks[i]])
                sklearn_data[f'{split}_y'] = convert(data.y[regular_masks[i]])
    
    # Optimisation: Pre-calculate batch sizes and NeighborLoaders OUTSIDE the seed loop
    train_loader, val_loader, test_loader = None, None, None
    alpha_focal = None
    batch_loader_datasets = ["IBM_AML_HiMedium", "IBM_AML_LiMedium"]
    detailed_metrics = []
    alpha_focal = balanced_class_weights(sklearn_data['train_y']) if model_name in sklearn_models or model_name in gpu_sklearn_models else balanced_class_weights(data.y[regular_masks[0]])

    
    use_loader = dataset_name in batch_loader_datasets
    
    seeds = [secrets.randbits(32) for _ in range(n_runs)]
    
    train_loader, val_loader, test_loader = None, None, None
    
    #Elliptic will never use loader so no need to account for elliptic masks
    if use_loader:
        from utilities import load_batch_size_by_phase
        num_neighbors = [10, 10]

        # Evaluation batch size
        print(f"Finding optimal EVALUATION batch size for {model_name}...")
        eval_batch_size = load_batch_size_by_phase(dataset_name, model_name, phase='evaluation')

        if eval_batch_size is None:
            def model_builder_for_eval_size_check():
                class MockTrial:
                    def suggest_int(self, name, low, high, step=None): return high
                    def suggest_float(self, name, low, high, log=False): return low
                    def suggest_categorical(self, name, choices): return choices[0]
                return _get_model_instance(MockTrial(), model_name, data, device, train_mask=regular_masks[0])

            eval_batch_size = find_optimal_batch_size(
                model_builder_for_eval_size_check, data, device, regular_masks[0],
                num_neighbors=num_neighbors, dataset_name=dataset_name,
                model_name=model_name, phase='evaluation'
            )
        print(f"  > Using EVALUATION batch size {eval_batch_size} for {model_name}")

        # Tuning batch size
        tuning_batch_size = load_batch_size_by_phase(dataset_name, model_name, phase='tuning') or 65536
        if tuning_batch_size == 65536:
            print(f"  > Using default TUNING batch size 65536 (no cached value found)")

        train_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=tuning_batch_size, input_nodes=regular_masks[0])
        val_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=eval_batch_size, input_nodes=regular_masks[1])
        test_loader = NeighborLoader(data, num_neighbors=num_neighbors, batch_size=eval_batch_size, input_nodes=regular_masks[2])
    else:
        print(f"  > Using FULL BATCH for {model_name} on {dataset_name} (no NeighborLoader)")
        
        
    detailed_metrics = []
    for i, seed in enumerate(seeds):
        set_seed(seed)
        print(f" > Run {i+1}/{n_runs} (Seed {seed})")
        
        fixed_trial = optuna.trial.FixedTrial(best_params)
        model_instance = _get_model_instance(fixed_trial, model_name, data, device, train_mask=regular_masks[0])
        
        y_pred, y_probs = None, None
        run_metrics = {}
        
        
    #I am here
        
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