# CLAUDE.md

## Project Overview

AML (Anti-Money Laundering) detection system using Graph Neural Networks (GNNs) and traditional ML models. Performs hyperparameter optimization via Optuna on financial transaction graph datasets to classify illicit vs. legitimate transactions.

## Key Commands

```bash
python3 main.py                  # Run hyperparameter tuning
python3 testing.py               # Run final model evaluation
python3 pre_process_datasets.py  # Preprocess datasets
```

Environment: Conda (Python 3.11.7), GPU-accelerated with CUDA (auto-fallback to CPU).

## Project Structure

```
main.py                    # Entry point for hyperparameter tuning
testing.py                 # Final evaluation pipeline
pre_process_datasets.py    # Data preprocessing orchestration
models.py                  # Model definitions (GCN, GAT, GIN, MLP, ModelWrapper)
dependencies.py            # Centralized imports
utilities.py               # GPU config, FocalLoss, seed setting, batch size caching
helper_functions.py        # Model instantiation, metrics, optimization utilities
training_functions.py      # Training loops and validation logic
funcs_for_optuna.py        # Optuna objective function and tuning orchestration
pre_processing.py          # Dataset classes (EllipticDataset, IBMAMLDataset, AMLSimDataset)
batch_size_cache.json      # Cached optimal batch sizes
optimization_results_*.db  # Optuna trial results (SQLite)
Datasets/                  # External data (gitignored)
```

## Models

- **Graph models**: GCN, GAT, GIN (PyTorch Geometric)
- **Non-graph**: MLP, XGBoost, SVM, Random Forest (scikit-learn)
- All GNN/MLP models use `ModelWrapper` for uniform training interface

## Datasets

- **Elliptic**: Bitcoin transaction graph (~203K nodes, time-step-based splits, special mask handling)
- **IBM AML**: HiSmall, LiSmall, HiMedium, LiMedium (synthetic, varying illicit ratios)
- **AMLSim**: Large-scale synthetic (~1.3M nodes)

Elliptic has unique `*_perf_eval_mask` variants for evaluating only on known-label nodes.

## Key Conventions

- snake_case for functions/variables, PascalCase for classes
- `_` prefix for internal/helper functions
- Aggressive GPU memory management: `torch.cuda.empty_cache()`, `gc.collect()`, strategic `del`
- FocalLoss used to handle class imbalance
- Optuna trials: 100 for GNNs, 50 for sklearn models
- Primary optimization metric: F1-illicit (validation set)
- Bug fixes documented inline with numbered comments (e.g., `# Bug 19 fix: ...`)

## Dependencies

PyTorch, PyTorch Geometric, Optuna, XGBoost, scikit-learn, torchmetrics, pandas, NumPy, matplotlib. No requirements.txt — managed via Conda.

## Important Notes

- Elliptic dataset requires special handling throughout (different mask structure, separate train/evaluate methods)
- Batch size auto-tuning uses binary search with caching per dataset-model-phase
- GPU memory limited to 95% of available to prevent system RAM spillover
- Results stored in SQLite databases (one per dataset)
