import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import os
import json


def configure_gpu_memory_limits(fraction=0.95, max_split_size_mb=512):
    """
    Configure GPU memory limits to prevent spillover to system RAM.
    
    Args:
        fraction: Maximum fraction of GPU memory to use (0.0 to 1.0)
        max_split_size_mb: Maximum size for memory splits in MB
    """
    if torch.cuda.is_available():
        # Limit the fraction of GPU memory PyTorch can use
        torch.cuda.set_per_process_memory_fraction(fraction)
        
        # Configure CUDA allocator for stricter memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
            f'max_split_size_mb:{max_split_size_mb},'
            f'garbage_collection_threshold:{fraction}'
        )
        print(f"GPU memory limited to {fraction*100}% of available memory")
        print(f"CUDA allocator configured with max_split_size_mb={max_split_size_mb}")



def save_batch_size_by_phase(dataset_name, model_name, batch_size, phase='tuning', cache_file='batch_size_cache.json'):
    """
    Save optimal batch size for a dataset-model combination with phase distinction.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        batch_size: Optimal batch size found
        phase: 'tuning' for hyperparameter tuning or 'evaluation' for final evaluation
        cache_file: Path to the cache file
    """
    cache = {}
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read {cache_file}, creating new cache")
    
    key = f"{dataset_name}_{model_name}_{phase}"
    cache[key] = batch_size
    
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)
    
    print(f"Saved {phase} batch size {batch_size} for {dataset_name}_{model_name}")


def load_batch_size_by_phase(dataset_name, model_name, phase='tuning', cache_file='batch_size_cache.json'):
    """
    Load cached batch size for a dataset-model combination with phase distinction.
    
    Args:
        dataset_name: Name of the dataset
        model_name: Name of the model
        phase: 'tuning' for hyperparameter tuning or 'evaluation' for final evaluation
        cache_file: Path to the cache file
        
    Returns:
        int or None: Cached batch size if found, None otherwise
    """
    if not os.path.exists(cache_file):
        return None
    
    try:
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        
        key = f"{dataset_name}_{model_name}_{phase}"
        batch_size = cache.get(key)
        
        if batch_size is not None:
            print(f"Loaded cached {phase} batch size {batch_size} for {dataset_name}_{model_name}")
            return batch_size
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load from {cache_file}: {e}")
    
    return None


def set_seed(seed: int):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            alpha_val = float(alpha)
            if not (0.0 <= alpha_val <= 1.0):
                raise ValueError("alpha float must lie in [0, 1]")
            self.alpha = torch.tensor([alpha_val, 1.0 - alpha_val], dtype=torch.float32)
        elif isinstance(alpha, (list, tuple, torch.Tensor)):
            self.alpha = torch.as_tensor(alpha, dtype=torch.float32)
        else:
            raise TypeError("alpha must be None, float, sequence, or torch.Tensor")

        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.ndim != 1:
                raise ValueError("alpha tensor must be 1-dimensional")
            if torch.any(self.alpha < 0):
                raise ValueError("alpha tensor must be non-negative")
            if self.alpha.sum() == 0:
                raise ValueError("alpha tensor must have positive sum")
            self.alpha = self.alpha / self.alpha.sum()

        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        logits = inputs.float()
        targets = targets.long()
        
        # Validation: Check number of classes matches logits output dimension
        num_classes = logits.shape[-1]
        
        # Validation: Check targets are in valid range [0, num_classes-1]
        if targets.min() < 0 or targets.max() >= num_classes:
            raise ValueError(
                f"Target values must be in range [0, {num_classes-1}], "
                f"but got min={targets.min().item()}, max={targets.max().item()}"
            )
        
        # Validation: Check alpha dimension matches num_classes
        if self.alpha is not None and len(self.alpha) != num_classes:
            raise ValueError(
                f"Alpha dimension ({len(self.alpha)}) must match number of classes ({num_classes})"
            )
            
        # Use log_softmax + nll_loss for numerical stability
        log_probs = F.log_softmax(logits, dim=-1)
        ce_loss = F.nll_loss(log_probs, targets, reduction='none')
        
        # More stable computation of pt
        pt = torch.exp(-ce_loss).clamp(min=1e-7, max=1.0-1e-7)  # Prevent underflow/overflow
        
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply alpha correctly depending on type
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]  # per-class weights
            focal_loss = at * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    
def extract_data_information(data):
    # Extracts masks from data object and recreates new data object to ensure no unnecessary attributes are included
    
    #Extract masks
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    y = data.y
    x = data.x
    edge_index = data.edge_index
    del data
    #Recreate data object
    new_data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )
    return new_data, train_mask, val_mask, test_mask


def verify_xgboost_gpu_support():
    """
    Verify XGBoost GPU support and print configuration information.
    
    This function checks:
    1. If CUDA is available for PyTorch
    2. If XGBoost is properly configured for GPU
    3. Creates a test XGBoost model to verify GPU functionality
    
    Returns:
        bool: True if XGBoost GPU is working, False otherwise
    """
    print("\n" + "="*80)
    print("XGBoost GPU Configuration Check")
    print("="*80)
    
    try:
        import xgboost as xgb
        print(f"✓ XGBoost version: {xgb.__version__}")
    except ImportError:
        print("✗ XGBoost is not installed")
        return False
    
    # Check PyTorch CUDA
    if torch.cuda.is_available():
        print(f"✓ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        print("✗ PyTorch CUDA not available")
        return False
    
    # Test XGBoost GPU with a simple dataset
    try:
        from xgboost import XGBClassifier
        print("\nTesting XGBoost GPU functionality...")
        
        # Create a small test dataset
        test_X = np.random.rand(100, 10)
        test_y = np.random.randint(0, 2, 100)
        
        # Try to create and fit a GPU-enabled XGBoost model
        test_model = XGBClassifier(
            n_estimators=10,
            tree_method='hist',
            device='cuda',
            eval_metric='logloss'
        )
        
        test_model.fit(test_X, test_y, verbose=False)
        
        print("✓ XGBoost GPU test successful!")
        print(f"  Device used: {test_model.get_params()['device']}")
        print(f"  Tree method: {test_model.get_params()['tree_method']}")
        print("\n" + "="*80)
        print("XGBoost is properly configured to use GPU acceleration")
        print("="*80 + "\n")
        return True
        
    except Exception as e:
        print(f"✗ XGBoost GPU test failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have xgboost with GPU support installed:")
        print("   conda install -c conda-forge py-xgboost-gpu")
        print("   or: pip install xgboost (if CUDA toolkit is installed)")
        print("2. Verify CUDA toolkit is installed and matches PyTorch CUDA version")
        print("3. Check that tree_method='hist' is set (required for GPU)")
        print("="*80 + "\n")
        return False

def extract_and_remove_masks(data):
    """
    Extracts specific masks from a PyTorch Data object, removes them from the 
    object, and returns the modified object and the masks.
    """
    # List of all potential mask attribute names
    mask_keys = [
        'train_mask', 'val_mask', 'test_mask',
        'train_perf_eval_mask', 'val_perf_eval_mask', 'test_perf_eval_mask'
    ]
    
    extracted_masks = {}
    
    for key in mask_keys:
        # Check if the attribute exists in the data object
        if hasattr(data, key):
            # Extract and store the mask
            extracted_masks[key] = getattr(data, key)
            # Delete the attribute from the data object
            delattr(data, key)
        else:
            # key does not exist, set to None
            extracted_masks[key] = None
            
    return data, extracted_masks