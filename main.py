import os
import gc
from dependencies import *
from utilities import *
from helper_functions import check_study_existence
from funcs_for_optuna import hyperparameter_tuning
from testing import run_final_evaluation
import optuna
configure_gpu_memory_limits(fraction=0.95, max_split_size_mb=512)
seeded_run = False
if seeded_run:
    set_seed(42)
    print("Seeded run with seed 42")
else:
    seed = np.random.SeedSequence().generate_state(1)[0]
    set_seed(seed)
#%% Hyperparameter tuning

#Checking whether the code is running on the HPC or on my windows pc
import platform
if platform.system() == 'Linux':
    # Store all paths when running linux in a variable to be accessed later
    dataset_paths = {
        "Elliptic": 'Datasets/Elliptic_dataset',
        "IBM_AML_HiSmall": 'Datasets/IBM_AML_dataset/HiSmall',
        "IBM_AML_LiSmall": 'Datasets/IBM_AML_dataset/LiSmall',
        "IBM_AML_HiMedium": 'Datasets/IBM_AML_dataset/HiMedium',
        "IBM_AML_LiMedium": 'Datasets/IBM_AML_dataset/LiMedium',
        "AMLSim": 'Datasets/AMLSim_dataset'
    }
    import sys
    datasets = [sys.argv[1]] #Getting dataset variable from submit.sh script
    models = [sys.argv[2]] #Getting model variable from submit.sh script
    
else:
    dataset_paths = {
        "Elliptic": 'Datasets/Elliptic_dataset',
        "IBM_AML_HiSmall": 'Datasets/IBM_AML_dataset/HiSmall',
        "IBM_AML_LiSmall": 'Datasets/IBM_AML_dataset/LiSmall',
        "IBM_AML_HiMedium": 'Datasets/IBM_AML_dataset/HiMedium',
        "IBM_AML_LiMedium": 'Datasets/IBM_AML_dataset/LiMedium',
        "AMLSim": 'Datasets/AMLSim_dataset'
    }
    datasets = ["Elliptic"]
    models = ["GCN", "GAT", "GIN", "MLP", "XGB", "RF", "SVM"]

print(f"Starting batch processing for {len(datasets)} datasets: {', '.join(datasets)}")
print("=" * 80)

for idx, dataset in enumerate(datasets, 1):
    print(f"\n{'='*80}")
    print(f"Hyperparameter tuning for datasets {idx}/{len(datasets)}: {dataset}")
    print(f"{'='*80}\n")

    #Check if there is a missing study in the database
    if all(check_study_existence(model, dataset) for model in models):
        print(f"All studies for {dataset} are complete. Skipping to next dataset.")
        continue
    else:
        match dataset:
            case "Elliptic":
                from pre_processing import EllipticDataset
                data = EllipticDataset(root=dataset_paths["Elliptic"])[0]
                dataset = "Elliptic"
            case "IBM_AML_HiSmall":
                from pre_processing import IBMAMLDataset_HiSmall
                data = IBMAMLDataset_HiSmall(root=dataset_paths["IBM_AML_HiSmall"])[0]
                dataset = "IBM_AML_HiSmall"
            case "IBM_AML_LiSmall":
                from pre_processing import IBMAMLDataset_LiSmall
                data = IBMAMLDataset_LiSmall(root=dataset_paths["IBM_AML_LiSmall"])[0]
                dataset = "IBM_AML_LiSmall"
            case "IBM_AML_HiMedium":
                from pre_processing import IBMAMLDataset_HiMedium
                data = IBMAMLDataset_HiMedium(root=dataset_paths["IBM_AML_HiMedium"])[0]
                dataset = "IBM_AML_HiMedium"
            case "IBM_AML_LiMedium":
                from pre_processing import IBMAMLDataset_LiMedium
                data = IBMAMLDataset_LiMedium(root=dataset_paths["IBM_AML_LiMedium"])[0]
                dataset = "IBM_AML_LiMedium"
            case "AMLSim":
                from pre_processing import AMLSimDataset
                data = AMLSimDataset(root=dataset_paths["AMLSim"])[0]
                dataset = "AMLSim"
        print(f"Dataset {dataset} loaded successfully for hyperparameter tuning.")

        data, masks = extract_and_remove_masks(data)
        print(f"Masks extracted and removed from data variable")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        loader_datasets = {"AMLSim", "IBM_AML_HiMedium", "IBM_AML_LiMedium", "IBM_AML_HiSmall", "IBM_AML_LiSmall"} #All datasets except Elliptic use NeighborLoader for training, so all datasets except Elliptic are in this set.

        #Starting optimisation
        print(f"Starting hyperparameter optimization for {dataset} dataset...")

        # Move data to device only for full-batch datasets; NeighborLoader datasets
        # stay on CPU so the full graph doesn't consume GPU memory
        if dataset not in loader_datasets:
            data = data.to(device)
            print(f"Data moved to device: {device}")
        else:
            print(f"Data kept on CPU for NeighborLoader-based training ({dataset})")

        print(f"Starting hyperparameter optimization for {dataset} dataset...")

        model_parameters = hyperparameter_tuning(
            models=models,
            dataset_name=dataset,
            data=data,
            masks=masks
        )

        # Clean up tuning artifacts before evaluation
        del data, masks, model_parameters
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory cleared after {dataset}")

        print(f"\n{'='*80}")
        print(f"✓ Successfully completed tuning for dataset {idx}/{len(datasets)}: {dataset}")
        print(f"{'='*80}\n")

print(f"\n{'='*80}")
print(f"HYPERPARAMETER TUNING COMPLETE")
print(f"All {len(datasets)} datasets have been processed.")
print(f"{'='*80}")

#%% Testing to get performance metrics

for idx, dataset in enumerate(datasets, 1):
    print(f"\n{'='*80}")
    print(f"Final evaluation for dataset {idx}/{len(datasets)}: {dataset}")
    print(f"{'='*80}\n")

    # Load best parameters from Optuna databases
    model_parameters = {}
    all_found = True
    for model_name in models:
        study_name = f'{model_name}_optimization on {dataset} dataset'
        storage_url = f'sqlite:///optimization_results_on_{dataset}_{model_name}.db'
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            model_parameters[model_name] = [study.best_trial.params]
            print(f"Loaded best params for {model_name} (best F1-illicit: {study.best_value:.4f})")
        except KeyError:
            print(f"No completed study found for {model_name} on {dataset}, skipping.")
            all_found = False

    if not model_parameters:
        print(f"No model parameters found for {dataset}. Skipping evaluation.")
        continue

    # Load dataset
    match dataset:
        case "Elliptic":
            from pre_processing import EllipticDataset
            data = EllipticDataset(root=dataset_paths["Elliptic"])[0]
        case "IBM_AML_HiSmall":
            from pre_processing import IBMAMLDataset_HiSmall
            data = IBMAMLDataset_HiSmall(root=dataset_paths["IBM_AML_HiSmall"])[0]
        case "IBM_AML_LiSmall":
            from pre_processing import IBMAMLDataset_LiSmall
            data = IBMAMLDataset_LiSmall(root=dataset_paths["IBM_AML_LiSmall"])[0]
        case "IBM_AML_HiMedium":
            from pre_processing import IBMAMLDataset_HiMedium
            data = IBMAMLDataset_HiMedium(root=dataset_paths["IBM_AML_HiMedium"])[0]
        case "IBM_AML_LiMedium":
            from pre_processing import IBMAMLDataset_LiMedium
            data = IBMAMLDataset_LiMedium(root=dataset_paths["IBM_AML_LiMedium"])[0]
        case "AMLSim":
            from pre_processing import AMLSimDataset
            data = AMLSimDataset(root=dataset_paths["AMLSim"])[0]

    data, masks = extract_and_remove_masks(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader_datasets = {"AMLSim", "IBM_AML_HiMedium", "IBM_AML_LiMedium", "IBM_AML_HiSmall", "IBM_AML_LiSmall"} #All datasets except Elliptic use NeighborLoader for evaluation, so all datasets except Elliptic are in this set.

    if dataset not in loader_datasets:
        data = data.to(device)
        print(f"Data moved to device: {device}")
    else:
        print(f"Data kept on CPU for NeighborLoader-based evaluation ({dataset})")

    run_final_evaluation(
        models=list(model_parameters.keys()),
        model_parameters=model_parameters,
        data=data,
        data_for_optimisation=dataset,
        masks=masks
    )

    # Clean up GPU memory between datasets
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared after {dataset}")

    print(f"\n{'='*80}")
    print(f"✓ Successfully completed evaluation for dataset {idx}/{len(datasets)}: {dataset}")
    print(f"{'='*80}\n")

print(f"\n{'='*80}")
print(f"ALL PROCESSING COMPLETE")
print(f"{'='*80}")

