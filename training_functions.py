import torch
import gc

def train_and_validate_with_loader(
    model_wrapper,
    train_loader, # Updated to accept loaders
    val_loader,   
    num_epochs,
    patience=None,
    min_delta=0.0,
    log_early_stop=False
):
    metrics = {
        'accuracy': [], 'precision_weighted': [], 'precision_illicit': [],
        'recall': [], 'recall_illicit': [], 'f1': [], 'f1_illicit': [],
        'roc_auc': [], 'PRAUC': [], 'kappa': [] 
    }
    
    best_f1 = -1
    epochs_without_improvement = 0
    best_f1_model_wts = None
    
    for epoch in range(num_epochs):
        train_loss, f1_illicit = model_wrapper.train_step(train_loader)
    
        
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
    
    return best_f1_model_wts, best_f1

def train_and_validate(
    model_wrapper,
    data,
    masks,   
    num_epochs,
    patience=None,
    min_delta=0.0,
    log_early_stop=False
):
    metrics = {
        'accuracy': [], 'precision_weighted': [], 'precision_illicit': [],
        'recall': [], 'recall_illicit': [], 'f1': [], 'f1_illicit': [],
        'roc_auc': [], 'PRAUC': [], 'kappa': [] 
    }
    
    best_f1 = -1
    epochs_without_improvement = 0
    best_f1_model_wts = None
    train_mask = masks['train_mask'].to(data.y.device)
    for epoch in range(num_epochs):
        train_loss, f1_illicit = model_wrapper.train_step(data.x[train_mask], data.y[train_mask])
    
        
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
    
    return best_f1_model_wts, best_f1


def update_best_weights(model, best_f1, current_f1, best_f1_model_wts=None):
    if current_f1 > best_f1:
        best_f1 = current_f1
        # Clone weights and ensure they stay on CPU to free GPU memory
        best_f1_model_wts = {k: v.cpu().clone().detach() for k, v in model.state_dict().items()}
        # Explicitly delete old weights to prevent accumulation
        old_wts = best_f1_model_wts
    return best_f1, best_f1_model_wts