import os
from dependencies import *
from utilities import *
from helper_functions import check_study_existence
from funcs_for_optuna import hyperparameter_tuning
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
    datasets = ["IBM_AML_HiMedium"]
    models = ["GCN", "GAT", "GIN", "MLP", "SVM", "XGB", "RF"]

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
        loader_datasets = {"AMLSim", "IBM_AML_HiMedium", "IBM_AML_LiMedium"}

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

        # Clean up GPU memory between datasets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU memory cleared after {dataset}")

        print(f"\n{'='*80}")
        print(f"✓ Successfully completed dataset {idx}/{len(datasets)}: {dataset}")
        print(f"{'='*80}\n")

print(f"\n{'='*80}")
print(f"BATCH PROCESSING COMPLETE")
print(f"All {len(datasets)} datasets have been processed.")
print(f"{'='*80}")
#%% Testing to get performance metrics

