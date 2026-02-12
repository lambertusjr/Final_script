import os
import secrets
import torch
import numpy as np
import pandas as pd
import optuna
import gc
from sklearn.metrics import accuracy_score, f1_score
from helper_functions import (
    _get_model_instance, balanced_class_weights, find_optimal_batch_size,
    calculate_metrics, calculate_pr_metrics_batched, save_pr_artifacts,
    save_metrics_to_pickle, save_dataframe_to_pickle
)
from training_functions import train_and_validate, train_and_validate_with_loader
from utilities import FocalLoss, set_seed, load_batch_size_by_phase
from models import ModelWrapper
from torch_geometric.loader import NeighborLoader


def run_final_evaluation(models, model_parameters, data, data_for_optimisation, masks):
    """
    Entry point: iterates over models and dispatches to evaluate_model_performance.
    """
    print(f"\nStarting FINAL EVALUATION for {data_for_optimisation} dataset...")

    # Create output directories once
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
    """
    Runs n_runs iterations of a single model+dataset combination, collects
    detailed per-run metrics, and saves summary statistics.

    Handles three model families:
        - wrapper_models  (MLP, GCN, GAT, GIN)  → PyTorch training loop
        - sklearn_models  (SVM, RF)              → scikit-learn .fit / .predict
        - gpu_sklearn_models (XGB)               → XGBoost on GPU tensors

    And three data-loading strategies:
        - NeighborLoader   (IBM_AML_HiMedium / LiMedium)
        - Full-batch       (AMLSim, IBM_AML_HiSmall / LiSmall)
        - Full-batch Elliptic (special mask handling)
    """

    # ── 0. Constants & device ────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wrapper_models  = {'MLP', 'GCN', 'GAT', 'GIN'}
    sklearn_models  = {'SVM', 'RF'}
    gpu_sklearn_models = {'XGB'}
    batch_loader_datasets = {"IBM_AML_HiMedium", "IBM_AML_LiMedium"}

    # ── 1. Validate masks vs data ────────────────────────────────────────
    # masks is a dict from extract_and_remove_masks:
    #   {'train_mask', 'val_mask', 'test_mask',
    #    'train_perf_eval_mask', 'val_perf_eval_mask', 'test_perf_eval_mask'}
    regular_masks = [masks["train_mask"], masks["val_mask"], masks["test_mask"]]
    train_mask, val_mask, test_mask = regular_masks

    is_elliptic = (dataset_name == "Elliptic")

    # ── 2. Prepare sklearn data (lazy, only if needed) ───────────────────
    sklearn_data = None
    if model_name in sklearn_models or model_name in gpu_sklearn_models:
        needs_cpu = model_name in sklearn_models  # SVM/RF need numpy
        convert = lambda t: t.cpu().numpy() if needs_cpu else t

        splits = ['train', 'val', 'test']
        sklearn_data = {}
        for i, split in enumerate(splits):
            sklearn_data[f'{split}_x'] = convert(data.x[regular_masks[i]])
            sklearn_data[f'{split}_y'] = convert(data.y[regular_masks[i]])
            if is_elliptic:
                perf_masks = [masks['train_perf_eval_mask'],
                              masks['val_perf_eval_mask'],
                              masks['test_perf_eval_mask']]
                sklearn_data[f'{split}_perf_x'] = convert(data.x[perf_masks[i]])
                sklearn_data[f'{split}_perf_y'] = convert(data.y[perf_masks[i]])

    # ── 3. Class weights for focal loss ──────────────────────────────────
    if sklearn_data is not None:
        # sklearn_data['train_y'] may be numpy (SVM/RF); balanced_class_weights expects a tensor
        alpha_focal = balanced_class_weights(torch.as_tensor(sklearn_data['train_y'], dtype=torch.long))
    else:
        alpha_focal = balanced_class_weights(data.y[train_mask])
    alpha_focal = alpha_focal.to(device)

    # ── 4. Determine loading strategy ────────────────────────────────────
    use_loader = dataset_name in batch_loader_datasets
    train_loader, val_loader, test_loader = None, None, None

    if use_loader:
        num_neighbors = [10, 10]

        # --- Evaluation batch size ---
        print(f"Finding optimal EVALUATION batch size for {model_name}...")
        eval_batch_size = load_batch_size_by_phase(dataset_name, model_name, phase='evaluation')

        if eval_batch_size is None:
            def model_builder_for_eval_size_check():
                class MockTrial:
                    def suggest_int(self, name, low, high, step=None): return high
                    def suggest_float(self, name, low, high, log=False): return low
                    def suggest_categorical(self, name, choices): return choices[0]
                return _get_model_instance(MockTrial(), model_name, data, device,
                                           train_mask=train_mask)

            eval_batch_size = find_optimal_batch_size(
                model_builder_for_eval_size_check, data, device, train_mask,
                num_neighbors=num_neighbors, dataset_name=dataset_name,
                model_name=model_name, phase='evaluation'
            )
        print(f"  > Using EVALUATION batch size {eval_batch_size} for {model_name}")

        # --- Tuning batch size ---
        tuning_batch_size = load_batch_size_by_phase(dataset_name, model_name, phase='tuning') or 65536
        if tuning_batch_size == 65536:
            print(f"  > Using default TUNING batch size 65536 (no cached value found)")

        train_loader = NeighborLoader(data, num_neighbors=num_neighbors,
                                      batch_size=tuning_batch_size, input_nodes=train_mask)
        val_loader   = NeighborLoader(data, num_neighbors=num_neighbors,
                                      batch_size=eval_batch_size, input_nodes=val_mask)
        test_loader  = NeighborLoader(data, num_neighbors=num_neighbors,
                                      batch_size=eval_batch_size, input_nodes=test_mask)
    else:
        print(f"  > Using FULL BATCH for {model_name} on {dataset_name} (no NeighborLoader)")

    # ── 5. Prepare masks dict for train_and_validate (full-batch) ────────
    #    train_and_validate expects a dict with specific keys.
    #    For Elliptic it looks up 'train_mask', 'train_perf_mask', 'val_mask', 'val_perf_mask'.
    #    For other full-batch datasets it uses 'train_mask' and 'val_mask'.
    if not use_loader and model_name in wrapper_models:
        if is_elliptic:
            # train_and_validate reads: masks['train_mask'], masks['train_perf_mask'],
            #                           masks['val_mask'],   masks['val_perf_mask']
            training_masks_dict = {
                'train_mask':      masks['train_mask'],
                'train_perf_mask': masks['train_perf_eval_mask'],
                'val_mask':        masks['val_mask'],
                'val_perf_mask':   masks['val_perf_eval_mask'],
            }
        else:
            training_masks_dict = {
                'train_mask': masks['train_mask'],
                'val_mask':   masks['val_mask'],
            }

    # ── 6. Adjust n_runs for RF ──────────────────────────────────────────
    if model_name == 'RF':
        n_runs = 10
    seeds = [secrets.randbits(32) for _ in range(n_runs)]

    # ══════════════════════════════════════════════════════════════════════
    # ── 7. MAIN RUN LOOP ─────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    detailed_metrics = []

    for i, seed in enumerate(seeds):
        set_seed(seed)
        print(f"  > Run {i+1}/{n_runs} (Seed {seed})")

        fixed_trial = optuna.trial.FixedTrial(best_params)
        model_instance = _get_model_instance(fixed_trial, model_name, data, device,
                                             train_mask=train_mask)

        y_pred, y_probs = None, None
        run_metrics = {}

        # ── 7a. Wrapper models (MLP / GCN / GAT / GIN) ──────────────────
        if model_name in wrapper_models:
            learning_rate = best_params.get('learning_rate', 0.01)
            weight_decay  = best_params.get('weight_decay', 5e-4)
            gamma_focal   = best_params.get('gamma_focal', 2.0)
            patience      = best_params.get('early_stop_patience', 20)
            min_delta     = best_params.get('early_stop_min_delta', 1e-3)

            criterion = FocalLoss(alpha=alpha_focal, gamma=gamma_focal, reduction='mean')
            optimiser = torch.optim.Adam(model_instance.parameters(),
                                         lr=learning_rate, weight_decay=weight_decay)

            model_wrapper = ModelWrapper(model_instance, optimiser, criterion)
            model_wrapper.model.to(device)

            num_epochs = 50 if model_name == "MLP" else 150

            # ── TRAIN ────────────────────────────────────────────────────
            if use_loader:
                best_wts, _ = train_and_validate_with_loader(
                    model_wrapper, train_loader, val_loader,
                    num_epochs,
                    patience=patience, min_delta=min_delta, log_early_stop=False
                )
            else:
                best_wts, _ = train_and_validate(
                    model_wrapper, data, training_masks_dict,
                    num_epochs, dataset_name,
                    patience=patience, min_delta=min_delta, log_early_stop=False
                )

            # Load best weights found during training
            model_wrapper.model.load_state_dict(best_wts)

            # ── EVALUATE ─────────────────────────────────────────────────
            if use_loader:
                _, test_metrics = model_wrapper.evaluate_loader(test_loader)

            elif is_elliptic:
                # evaluate_elliptic expects [test_mask, test_perf_eval_mask]
                elliptic_test_masks = [masks['test_mask'],
                                       masks['test_perf_eval_mask']]
                _, test_metrics = model_wrapper.evaluate_elliptic(data, elliptic_test_masks)

            else:
                # evaluate_full expects a single mask
                _, test_metrics = model_wrapper.evaluate_full(data, test_mask)

            y_probs    = test_metrics['probs']
            y_pred     = test_metrics['preds']
            y_true     = test_metrics['y_true']
            run_metrics = test_metrics

        # ── 7b. Sklearn models (SVM / RF) ────────────────────────────────
        elif model_name in sklearn_models:
            if is_elliptic:
                train_x_key, train_y_key = 'train_perf_x', 'train_perf_y'
                test_x_key,  test_y_key  = 'test_perf_x',  'test_perf_y'
            else:
                train_x_key, train_y_key = 'train_x', 'train_y'
                test_x_key,  test_y_key  = 'test_x',  'test_y'

            model_instance.fit(sklearn_data[train_x_key], sklearn_data[train_y_key])
            y_true = sklearn_data[test_y_key]

            try:
                y_probs = model_instance.predict_proba(sklearn_data[test_x_key])
            except AttributeError:
                # Fallback for SVM: scale decision_function to [0, 1]
                if hasattr(model_instance, 'decision_function'):
                    dfunc = model_instance.decision_function(sklearn_data[test_x_key])
                    y_probs = (dfunc - dfunc.min()) / (dfunc.max() - dfunc.min())
                else:
                    y_probs = None

            y_pred = model_instance.predict(sklearn_data[test_x_key])

            if y_probs is not None:
                run_metrics = calculate_metrics(y_true, y_pred, y_probs)
            else:
                run_metrics = {
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': np.nan, 'precision_illicit': np.nan,
                    'recall': np.nan, 'recall_illicit': np.nan,
                    'f1': f1_score(y_true, y_pred, average='weighted'),
                    'f1_illicit': f1_score(y_true, y_pred, pos_label=1, average='binary'),
                    'roc_auc': np.nan, 'PRAUC': np.nan, 'kappa': np.nan,
                }

        # ── 7c. GPU sklearn models (XGB) ────────────────────────────────
        elif model_name in gpu_sklearn_models:
            if is_elliptic:
                train_x_key, train_y_key = 'train_perf_x', 'train_perf_y'
                test_x_key,  test_y_key  = 'test_perf_x',  'test_perf_y'
            else:
                train_x_key, train_y_key = 'train_x', 'train_y'
                test_x_key,  test_y_key  = 'test_x',  'test_y'

            model_instance.fit(sklearn_data[train_x_key], sklearn_data[train_y_key])
            y_true = sklearn_data[test_y_key]

            y_probs = model_instance.predict_proba(sklearn_data[test_x_key])
            y_pred  = model_instance.predict(sklearn_data[test_x_key])
            run_metrics = calculate_metrics(
                y_true.cpu().numpy() if hasattr(y_true, 'cpu') else y_true,
                y_pred.cpu().numpy() if hasattr(y_pred, 'cpu') else y_pred,
                y_probs.cpu().numpy() if hasattr(y_probs, 'cpu') else y_probs
            )

        # ── 8. PR curve computation (shared across all models) ───────────
        if y_probs is not None:
            # Ensure numpy arrays for consistency
            y_probs_np = y_probs.cpu().numpy() if hasattr(y_probs, 'cpu') else (
                y_probs if isinstance(y_probs, np.ndarray) else np.array(y_probs))
            y_true_np = y_true.cpu().numpy() if hasattr(y_true, 'cpu') else (
                y_true if isinstance(y_true, np.ndarray) else np.array(y_true))

            # Extract positive-class probabilities
            y_probs_for_pr = y_probs_np[:, 1] if y_probs_np.ndim == 2 else y_probs_np

            y_probs_tensor = torch.as_tensor(y_probs_for_pr, dtype=torch.float32)
            y_true_tensor  = torch.as_tensor(y_true_np, dtype=torch.long)

            precision, recall, thresholds = calculate_pr_metrics_batched(
                y_probs_tensor, y_true_tensor)

            pr_filename = f"results/{dataset_name}/pr_curves/{model_name}_run_{i+1}"
            pr_auc = save_pr_artifacts(precision, recall, thresholds, pr_filename)
            run_metrics['PRAUC'] = pr_auc

            pr_data = {
                'precision':  precision.numpy(),
                'recall':     recall.numpy(),
                'thresholds': thresholds.numpy(),
                'auc':        pr_auc,
                'y_true':     y_true_np,
                'y_probs':    y_probs_for_pr,
            }
            save_metrics_to_pickle(
                pr_data,
                f"results/{dataset_name}/pkl_files/{model_name}_run_{i+1}_pr_data.pkl"
            )

        # ── 9. Tag and store run metrics ─────────────────────────────────
        run_metrics['model'] = model_name
        run_metrics['run']   = i + 1
        detailed_metrics.append(run_metrics)

        save_metrics_to_pickle(
            run_metrics,
            f"results/{dataset_name}/pkl_files/{model_name}_run_{i+1}_metrics.pkl"
        )

        # ── 10. Per-run cleanup ──────────────────────────────────────────
        del model_instance, y_pred, y_probs, run_metrics
        if model_name in wrapper_models:
            del model_wrapper, best_wts
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ══════════════════════════════════════════════════════════════════════
    # ── 11. AGGREGATE & SAVE ─────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════
    df = pd.DataFrame(detailed_metrics)
    df.to_csv(f"results/{dataset_name}/metrics/{model_name}_detailed_metrics.csv", index=True)
    save_dataframe_to_pickle(df, f"results/{dataset_name}/pkl_files/{model_name}_detailed_metrics.pkl")

    columns_to_drop = {'model', 'run', 'probs', 'preds', 'y_true'}
    numeric_df = df.drop(columns=[col for col in columns_to_drop if col in df.columns],
                         errors='ignore')
    summary = numeric_df.agg(['mean', 'std']).transpose()
    summary['model'] = model_name

    save_dataframe_to_pickle(summary, f"results/{dataset_name}/pkl_files/{model_name}_summary_metrics.pkl")

    summary_file = f"results/{dataset_name}/metrics/summary_metrics.csv"
    summary.to_csv(summary_file,
                   mode='a' if os.path.exists(summary_file) else 'w',
                   header=not os.path.exists(summary_file))

    print(f"  > Completed {model_name}. Metrics saved to results/{dataset_name}/metrics/")
