import torch
import gc
from utilities import ram_is_critical, check_ram_usage, vram_is_critical, check_vram_usage

def train_and_validate_with_loader(
    model_wrapper,
    train_loader, # Updated to accept loaders
    val_loader,
    num_epochs,
    patience=None,
    min_delta=0.0,
    log_early_stop=False
):
    best_f1 = -1
    epochs_without_improvement = 0
    best_f1_model_wts = None
    
    for epoch in range(num_epochs):
        # Bug 11 fix: method is train_step_loader, not train_step
        train_loss, f1_illicit = model_wrapper.train_step_loader(train_loader)

        # Bug 12 fix: validate on val_loader each epoch so early stopping uses val F1, not train F1
        _, val_f1_illicit = model_wrapper.evaluate_loader_mini(val_loader)
        current_f1 = val_f1_illicit
        
        improved = current_f1 > (best_f1 + min_delta)
        if improved:
            best_f1, best_f1_model_wts = update_best_weights(model_wrapper.model, best_f1, current_f1, best_f1_model_wts)
            epochs_without_improvement = 0
            best_epoch = epoch
        else:
            epochs_without_improvement += 1
        
        if patience and epochs_without_improvement >= patience:
            if log_early_stop:
                print(f"Early stopping at epoch {epoch+1}. Best F1-illicit: {best_f1:.4f} at epoch {best_epoch+1}.")
            break
        
        # Periodic aggressive cleanup to prevent memory accumulation during long training runs
        if (epoch + 1) % 25 == 0:  # Every 25 epochs
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            if ram_is_critical(threshold=0.85):
                usage_pct, avail_gb = check_ram_usage()
                print(f"WARNING: RAM usage {usage_pct:.1f}% at epoch {epoch+1}, "
                      f"only {avail_gb:.1f} GB available. "
                      f"System may start paging to SSD.")
            if torch.cuda.is_available() and vram_is_critical(threshold=0.92):
                vram_frac, vram_free = check_vram_usage()
                print(f"WARNING: VRAM usage {vram_frac*100:.1f}% at epoch {epoch+1}, "
                      f"only {vram_free:.2f} GB free.")

    return best_f1_model_wts, best_f1

def train_and_validate(
    model_wrapper,
    data,
    masks,
    num_epochs,
    dataset_name,
    patience=None,
    min_delta=0.0,
    log_early_stop=False
):
    #List for which datasets can run on full batching loop without running out of memory. For these datasets, we will not use neighbourloader and will train on the full graph.
    full_batch_datasets = ["IBM_AML_HiSmall", "IBM_AML_LiSmall"] #Elliptic is also included but uses a different training loop.
    
    best_f1 = -1
    epochs_without_improvement = 0
    best_f1_model_wts = None
    masks = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in masks.items()}
    #extracting masks into training and validation groups
    if dataset_name == "Elliptic":
        # Bug 10 fix: keys are 'train_perf_eval_mask'/'val_perf_eval_mask' (from utilities.py), not 'train_perf_mask'/'val_perf_mask'
        elliptic_training_masks = [masks['train_mask'], masks['train_perf_eval_mask']]
        elliptic_validation_masks = [masks['val_mask'], masks['val_perf_eval_mask']]

    for epoch in range(num_epochs):
        #Train step and evaluation step to determine best weights.
        if dataset_name == "Elliptic":
            model_wrapper.train_step_elliptic(data, elliptic_training_masks)
            loss, f1_illicit = model_wrapper.mini_eval_elliptic(data, elliptic_validation_masks)
            
        elif dataset_name in full_batch_datasets:
            model_wrapper.train_step_full(data, masks['train_mask'])
            loss, f1_illicit = model_wrapper.mini_eval_full(data, masks['val_mask']) #Passing val_mask for evaluation 

        
        current_f1 = f1_illicit
        
        improved = current_f1 > (best_f1 + min_delta)
        if improved:
            best_f1, best_f1_model_wts = update_best_weights(model_wrapper.model, best_f1, current_f1, best_f1_model_wts)
            epochs_without_improvement = 0
            best_epoch = epoch
        else:
            epochs_without_improvement += 1
        
        if patience and epochs_without_improvement >= patience:
            if log_early_stop:
                print(f"Early stopping at epoch {epoch+1}. Best F1-illicit: {best_f1:.4f} at epoch {best_epoch+1}.")
            break
        
        # Periodic aggressive cleanup to prevent memory accumulation during long training runs
        if (epoch + 1) % 25 == 0:  # Every 25 epochs
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            if ram_is_critical(threshold=0.85):
                usage_pct, avail_gb = check_ram_usage()
                print(f"WARNING: RAM usage {usage_pct:.1f}% at epoch {epoch+1}, "
                      f"only {avail_gb:.1f} GB available. "
                      f"System may start paging to SSD.")
            if torch.cuda.is_available() and vram_is_critical(threshold=0.92):
                vram_frac, vram_free = check_vram_usage()
                print(f"WARNING: VRAM usage {vram_frac*100:.1f}% at epoch {epoch+1}, "
                      f"only {vram_free:.2f} GB free.")

    return best_f1_model_wts, best_f1


def update_best_weights(model, best_f1, current_f1, best_f1_model_wts=None):
    if current_f1 > best_f1:
        best_f1 = current_f1
        # Clone weights and ensure they stay on CPU to free GPU memory
        best_f1_model_wts = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
    return best_f1, best_f1_model_wts