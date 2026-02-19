import torch
import numpy as np
from torchmetrics.classification import BinaryPrecisionRecallCurve
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score, auc
from torchmetrics.classification import BinaryAveragePrecision
import matplotlib.pyplot as plt
import pickle
import os
import optuna
metric = BinaryAveragePrecision().to('cuda') if torch.cuda.is_available() else BinaryAveragePrecision()

from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
import gc
import time
from contextlib import contextmanager
from torch_geometric.loader import NeighborLoader



def calculate_pr_metrics_batched(probs, labels, chunk_size=10000):
    # 1. Initialize metric on CPU to save GPU memory
    #    thresholds=None calculates exact curve (uses more RAM)
    #    thresholds=1000 uses fixed bins (uses constant minimal RAM)
    pr_curve = BinaryPrecisionRecallCurve(thresholds=None).cpu()

    n_samples = probs.size(0)

    # 2. Iterate manually in chunks (simulating batches)
    with torch.no_grad():
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            
            # SLICE AND MOVE TO CPU IMMEDIATELY
            # Essential: .detach() breaks the graph, .cpu() frees VRAM
            prob_chunk = probs[i:end].detach().cpu()
            label_chunk = labels[i:end].detach().cpu()
            
            # Accumulate stats
            pr_curve.update(prob_chunk, label_chunk)

    # 3. Compute final metric (on CPU)
    #    returns precision, recall, thresholds
    precision, recall, thresholds = pr_curve.compute()
    
    # Optional: Plotting
    # fig, ax = pr_curve.plot(score=True) 
    
    return precision, recall, thresholds

# Usage Example:
# precision, recall, _ = calculate_pr_metrics_batched(out, data.y)

def save_pr_artifacts(precision, recall, thresholds, filename_prefix):
    """
    Takes PRE-CALCULATED metrics and saves them to disk.
    Does NOT perform calculation.
    """
    # 1. Calculate AUC (Cheap)
    pr_auc = auc(recall.numpy(), precision.numpy())
    
    # 2. Save Raw Data
    np.savez_compressed(
        f"{filename_prefix}_pr_data.npz",
        precision=precision.numpy(),
        recall=recall.numpy(),
        thresholds=thresholds.numpy(),
        auc=pr_auc
    )

    # 3. Save Image
    try:
        fig, ax = plt.subplots()
        ax.plot(recall.numpy(), precision.numpy(), label=f'PRAUC = {pr_auc:.4f}')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.legend()
        fig.savefig(f"{filename_prefix}_pr_curve.png", dpi=300)
    finally:
        plt.close(fig) # Prevent memory leak
        
    return pr_auc

def save_metrics_to_pickle(metrics_dict, filename):
    """
    Save metrics dictionary to a pickle file.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary containing metrics to save.
    filename : str
        Full path to the pickle file (including .pkl extension).
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(metrics_dict, f)
    print(f"Saved metrics to {filename}")

def save_dataframe_to_pickle(df, filename):
    """
    Save pandas DataFrame to a pickle file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filename : str
        Full path to the pickle file (including .pkl extension).
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
    print(f"Saved DataFrame to {filename}")

def load_pickle(filename):
    """
    Load data from a pickle file.
    
    Parameters
    ----------
    filename : str
        Full path to the pickle file.
        
    Returns
    -------
    data : Any
        The loaded data (dict, DataFrame, etc.)
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def calculate_metrics(y_true, y_pred, y_pred_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_illicit = precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0) # illicit is class 1
    recall = recall_score(y_true, y_pred, average='weighted')
    recall_illicit = recall_score(y_true, y_pred, pos_label=1, average='binary') # illicit is class 1
    f1 = f1_score(y_true, y_pred, average='weighted')
    f1_illicit = f1_score(y_true, y_pred, pos_label=1, average='binary') # illicit is class 1
    
    prob_positive = y_pred_prob[:, 1] if y_pred_prob.ndim == 2 else y_pred_prob
    roc_auc = roc_auc_score(y_true, prob_positive)  # assuming class 1 is the positive class
    pr_auc = np.nan
    
    kappa = cohen_kappa_score(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'precision_illicit': precision_illicit,
        'recall': recall,
        'recall_illicit': recall_illicit,
        'f1': f1,
        'f1_illicit': f1_illicit,
        'roc_auc': roc_auc,
        'PRAUC': pr_auc,
        'kappa': kappa,
    }
    
    return metrics

def balanced_class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """
    Compute inverse-frequency class weights (sum to 1) for 1-D integer labels.

    Unlabelled entries (label < 0) are ignored.
    """
    if labels.ndim != 1:
        labels = labels.view(-1)
    labels = labels.detach().cpu()  # Ensure on CPU
    valid = labels >= 0
    if not torch.any(valid):
        return torch.ones(num_classes, dtype=torch.float32) / float(num_classes)
    filtered = labels[valid].to(torch.long)
    counts = torch.bincount(filtered, minlength=num_classes).clamp_min(1)
    inv = (1.0 / counts.float())
    inv = inv / inv.sum()
    return inv

def _get_model_instance(trial, model, data, device, train_mask=None):
    """
    Helper function to suggest hyperparameters and instantiate a model
    based on the model name.
    """
    if model == 'MLP':
        from models import MLP
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.7)
        dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.7)
        return MLP(num_node_features=data.num_node_features, num_classes=2, hidden_units=hidden_units, dropout_1=dropout_1, dropout_2=dropout_2)
    
    elif model == 'SVM':
        C = trial.suggest_float('C', 0.01, 100.0, log=True)
        return SGDClassifier(
            loss='hinge',
            alpha=1.0 / (C * data.num_nodes),  # Convert C to alpha
            penalty='l2',
            max_iter=1000,
            tol=1e-3,
            class_weight='balanced',
            random_state=42
        )

    elif model == 'XGB':
        max_depth = trial.suggest_int('max_depth', 5, 15)
        Gamma_XGB = trial.suggest_float('Gamma_XGB', 0, 5)
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        learning_rate_XGB = trial.suggest_float('learning_rate_XGB', 0.001, 0.3, log=True) # XGB learning rate
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
        reg_alpha = trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
        return XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=calculate_scale_pos_weight(data, train_mask),
            learning_rate=learning_rate_XGB,
            max_depth=max_depth,
            n_estimators=n_estimators,
            colsample_bytree=colsample_bytree,
            gamma=Gamma_XGB,
            subsample=subsample,
            min_child_weight=min_child_weight,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            tree_method='hist',
            device="cpu"
        )

    elif model == 'RF':
        from sklearn.ensemble import RandomForestClassifier
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        max_depth = trial.suggest_int('max_depth', 5, 15)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2'])
        return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                      max_features=max_features, class_weight='balanced', n_jobs=-1)

    elif model == 'GCN':
        from models import GCN
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        dropout = trial.suggest_float('dropout', 0.0, 0.7)
        return GCN(num_node_features=data.x.shape[1], num_classes=2, hidden_units=hidden_units, dropout=dropout)

    elif model == 'GAT':
        from models import GAT
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        num_heads = trial.suggest_int('num_heads', 1, 8)
        dropout_1 = trial.suggest_float('dropout_1', 0.0, 0.7)
        dropout_2 = trial.suggest_float('dropout_2', 0.0, 0.7)
        return GAT(num_node_features=data.x.shape[1], num_classes=2, hidden_units=hidden_units, num_heads=num_heads, dropout_1=dropout_1, dropout_2=dropout_2)

    elif model == 'GIN':
        from models import GIN
        hidden_units = trial.suggest_int('hidden_units', 32, 256)
        dropout = trial.suggest_float('dropout', 0.0, 0.5)
        return GIN(num_node_features=data.x.shape[1], num_classes=2, hidden_units=hidden_units, dropout=dropout)

    else:
        raise ValueError(f"Unknown model: {model}")
    
def calculate_scale_pos_weight(data, train_mask):
    """
    Calculate the scale_pos_weight for imbalanced datasets.
    """
    if train_mask is None:
        raise ValueError("train_mask must be provided to calculate_scale_pos_weight to prevent data leakage.")
        
    y_train = data.y[train_mask].cpu().numpy()
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    return float(neg) / float(pos)

def check_study_existence(model_name, data_for_optimization):
    """
    Check if an Optuna study exists and has a sufficient number of trials (>= 50).
    
    If a study exists but has fewer than 50 trials, it is automatically
    deleted.
    
    Parameters
    ----------
    model_name : str
        Name of the model (MLP, SVM, XGB, RF, GCN, GAT, GIN).
    data_for_optimization : str
        Name of the dataset used for optimization.
        
    Returns
    -------
    exists : bool
        True if a study exists with >= 50 trials, False otherwise.
    """
    study_name = f'{model_name}_optimization on {data_for_optimization} dataset'
    storage_url = f'sqlite:///optimization_results_on_{data_for_optimization}_{model_name}.db'

    
    try:
        # 1. Attempt to load the study
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        # 2. Study exists, check the number of trials (runs)
        num_trials = len(study.trials)
        
        if num_trials < 35:
            # 3. Less than 50 runs: wipe the study and return False
            print(f"Study '{study_name}' found with only {num_trials} trials (< 50). Deleting study.")
            optuna.delete_study(study_name=study_name, storage=storage_url)
            return False
        else:
            # 4. 50 or more runs: study is valid, return True
            print(f"Study '{study_name}' found with {num_trials} trials (>= 50). Study is valid.")
            return True
            
    except KeyError:
        # 5. Study does not exist: return False
        print(f"Study '{study_name}' not found.")
        return False
    
def run_trial_with_cleanup(trial_func, model_name, *args, **kwargs):
    """
    Runs a trial function safely with:
      - Automatic no_grad() for CPU-based models.
      - GPU/CPU memory cleanup after each trial.
      - Aggressive memory reclamation and cache clearing.
    
    Parameters
    ----------
    trial_func : callable
        The trial function to run (e.g., objective).
    model_name : str
        Name of the model (MLP, SVM, XGB, RF, GCN, GAT, GIN).
    *args, **kwargs :
        Arguments to pass to trial_func.
        
    Returns
    -------
    result : Any
        The return value of the trial function.
    """
    try:
        with inference_mode_if_needed(model_name):
            result = trial_func(*args, **kwargs)
        return result
    finally:
        # Aggressive cleanup after each trial
        if torch.cuda.is_available():
            # Synchronize to ensure all GPU operations complete
            torch.cuda.synchronize()
            # Multiple rounds of cache clearing to address fragmentation
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        
        # Aggressive garbage collection
        gc.collect()
        gc.collect()  # Run twice to catch circular references
        
        # Force collection of unreachable objects
        import sys
        if hasattr(sys, 'ps1'):  # Only if interactive session
            pass  # Don't pollute notebook output
        else:
            # In script mode, additional cleanup
            gc.collect()
        
@contextmanager
def inference_mode_if_needed(model_name: str):
    """
    Context manager that disables gradient tracking if the model is CPU-based
    or if we are in evaluation mode.
    """
    if model_name in ["SVM", "XGB", "RF"]:
        with torch.no_grad():
            yield
    else:
        yield

def find_optimal_batch_size(model_builder, data, device, train_mask, num_neighbors=[10, 10], dataset_name=None, model_name=None, phase='tuning'):
    """
    Finds the optimal batch size for NeighborLoader by testing increasing sizes
    until OOM, then binary searching. Prevents spillover to system RAM.
    Caches results for future use.
    
    Args:
        model_builder: A function that returns a fresh model instance (on CPU).
        data: The data object.
        device: The device to train on.
        train_mask: Mask for training nodes.
        num_neighbors: Neighbor sampling configuration.
        dataset_name: Name of dataset (for caching).
        model_name: Name of model (for caching).
        phase: 'tuning' for hyperparameter tuning or 'evaluation' for final evaluation.
               Evaluation phase will allocate more memory.
    
    Returns:
        int: Optimal batch size (approx 90% of max safe size).
    """
    # Check cache first
    if dataset_name and model_name:
        from utilities import load_batch_size_by_phase
        cached_batch_size = load_batch_size_by_phase(dataset_name, model_name, phase=phase)
        if cached_batch_size is not None:
            return cached_batch_size
    
    print(f"Searching for optimal {phase} batch size...")
    
    # Get total GPU memory to set hard limits
    if torch.cuda.is_available():
        total_gpu_memory = torch.cuda.get_device_properties(device).total_memory
        # For evaluation phase, use more aggressive memory allocation
        if phase == 'evaluation':
            reserved_fraction = 0.92  # Use up to 92% for evaluation
            print(f"Running in EVALUATION mode: will use up to 92% of GPU memory")
        else:
            reserved_fraction = 0.90  # More conservative for tuning (avoid OOM during training)
            print(f"Running in TUNING mode: will use up to 90% of GPU memory")
        
        max_memory_limit = int(total_gpu_memory * reserved_fraction)
        print(f"GPU memory limit: {max_memory_limit / (1024**3):.2f} GB") #1024^3 = GB
    else:
        max_memory_limit = None
    
    low = 32768  # Start higher for better GPU utilization
    high = 30000000000 # Increase upper bound
    optimal = 65536 # Higher safe default
    
    from utilities import ram_is_critical, check_ram_usage, vram_is_critical, check_vram_usage

    # Define a simple training loop for testing
    def test_batch_size(batch_size):
        # Guard: check RAM and VRAM before even attempting this batch size.
        usage_pct, avail_gb = check_ram_usage()
        if ram_is_critical(threshold=0.85):
            print(f"Batch size {batch_size} skipped: RAM already at {usage_pct:.1f}% "
                  f"({avail_gb:.1f} GB available) before test started.")
            return False

        if torch.cuda.is_available():
            vram_frac, vram_free = check_vram_usage()
            if vram_is_critical(threshold=0.90):
                print(f"Batch size {batch_size} skipped: VRAM already at {vram_frac*100:.1f}% "
                      f"({vram_free:.2f} GB free) before test started.")
                return False

        try:
            # Create a fresh model and move to device
            model = model_builder().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            criterion = torch.nn.CrossEntropyLoss()

            loader = NeighborLoader(
                data,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                input_nodes=train_mask,
                shuffle=True
            )

            model.train()
            # Run batches to ensure peak memory is reached, checking RAM on
            # every step so we abort as soon as paging would begin.
            steps = 0
            ram_exceeded = False
            for batch in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)

                # Slice logic similar to ModelWrapper
                batch_size_actual = batch.batch_size
                out_sliced = out[:batch_size_actual]
                y_sliced = batch.y[:batch_size_actual]

                loss = criterion(out_sliced, y_sliced)
                loss.backward()
                optimizer.step()

                steps += 1

                # Check RAM after every batch step so we catch pressure early
                usage_pct, avail_gb = check_ram_usage()
                if ram_is_critical(threshold=0.85):
                    ram_exceeded = True
                    print(f"Batch size {batch_size} rejected at step {steps}: "
                          f"RAM usage {usage_pct:.1f}% ({avail_gb:.1f} GB available). "
                          f"Reducing to prevent SSD paging.")
                    break

                # Check VRAM after every batch step
                if torch.cuda.is_available():
                    vram_frac, vram_free = check_vram_usage()
                    if vram_is_critical(threshold=0.90):
                        ram_exceeded = True
                        print(f"Batch size {batch_size} rejected at step {steps}: "
                              f"VRAM usage {vram_frac*100:.1f}% ({vram_free:.2f} GB free). "
                              f"Reducing to prevent GPU OOM.")
                        break

                if steps >= 10: # Test 10 batches for more accurate memory profile
                    break

            del model, optimizer, criterion, loader
            if steps > 0:
                del batch, out, out_sliced, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            return not ram_exceeded
            
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                print(f"Batch size {batch_size} failed with GPU error: {type(e).__name__}")
                if torch.cuda.is_available():
                     torch.cuda.empty_cache()
                return False
            else:
                print(f"Batch size {batch_size} failed with RuntimeError: {e}")
                if torch.cuda.is_available():
                     torch.cuda.empty_cache()
                return False
        except Exception as e:
            print(f"Batch size {batch_size} failed with error: {type(e).__name__}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

    # 1. Exponential search to find upper bound
    current = low
    max_safe = low
    
    while current <= high:
        print(f"Testing batch size: {current}")
        if test_batch_size(current):
            max_safe = current
            current *= 2
        else:
            high = current
            break
            
    # 2. Binary search between max_safe and high (which failed)
    low = max_safe
    
    while low < high - 2048: # Granularity of 2048
        mid = (low + high) // 2
        print(f"Binary search testing: {mid}")
        if test_batch_size(mid):
            low = mid
        else:
            high = mid
    
    # Return a percentage of max found size based on phase
    if phase == 'evaluation':
        # For evaluation: use 95% of max (more aggressive)
        optimal = int(low * 0.95)
        print(f"Optimal {phase} batch size found: {optimal}")
    else:
        # For tuning: use 90% of max (more conservative)
        optimal = int(low * 0.90)
        print(f"Optimal {phase} batch size found: {optimal}")
    
    # Save to cache if dataset and model names provided
    if dataset_name and model_name:
        from utilities import save_batch_size_by_phase
        save_batch_size_by_phase(dataset_name, model_name, optimal, phase=phase)
    
    return optimal

