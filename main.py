import os
import traceback
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
datasets = ["Elliptic", "IBM_AML_HiSmall", "IBM_AML_LiSmall"]
models = ["MLP", "SVM", "XGB", "RF", "GCN", "GAT", "GIN"]
print(f"Starting batch processing for {len(datasets)} datasets: {', '.join(datasets)}")
print("=" * 80)

for idx, dataset in enumerate(datasets, 1):
    print(f"\n{'='*80}")
    print(f"Hyperparameter tuning for datasets {idx}/{len(datasets)}: {dataset}")
    print(f"{'='*80}\n")

    #Check if there is a missing study in the database
    db_path = f'sqlite:///optimization_results_on_{dataset}.db'
    if all(check_study_existence(model, dataset) for model in models):
        print(f"All studies for {dataset} are complete. Skipping to next dataset.")
        continue
    else: 
        try:
            match dataset:
                case "Elliptic":
                    from pre_processing import EllipticDataset
                    data = EllipticDataset(root='Datasets/Elliptic_dataset')[0]
                    dataset = "Elliptic"
                case "IBM_AML_HiSmall":
                    from pre_processing import IBMAMLDataset_HiSmall
                    data = IBMAMLDataset_HiSmall(root='Datasets/IBM_AML_dataset/HiSmall')[0]
                    dataset = "IBM_AML_HiSmall"
                case "IBM_AML_LiSmall":
                    from pre_processing import IBMAMLDataset_LiSmall
                    data = IBMAMLDataset_LiSmall(root='Datasets/IBM_AML_dataset/LiSmall')[0]
                    dataset = "IBM_AML_LiSmall"
                case "IBM_AML_HiMedium":
                    from pre_processing import IBMAMLDataset_HiMedium
                    data = IBMAMLDataset_HiMedium(root='Datasets/IBM_AML_dataset/HiMedium')[0]
                    dataset = "IBM_AML_HiMedium"
                case "IBM_AML_LiMedium":
                    from pre_processing import IBMAMLDataset_LiMedium
                    data = IBMAMLDataset_LiMedium(root='Datasets/IBM_AML_dataset/LiMedium')[0]
                    dataset = "IBM_AML_LiMedium"
                case "AMLSim":
                    from pre_processing import AMLSimDataset
                    data = AMLSimDataset(root='Datasets/AMLSim_dataset')[0]
                    dataset = "AMLSim"
            print(f"Dataset {dataset} loaded successfully for hyperparameter tuning.")

            data, masks = extract_and_remove_masks(data)
            print(f"Masks extracted and removed from data variable")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            #Starting optimisation
            print(f"Starting hyperparameter optimization for {dataset} dataset...")

            # Move data to device
            data = data.to(device)
            print(f"Data moved to device: {device}")
            
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
            
        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"✗ ERROR processing dataset {dataset}:")
            print(f"  {type(e).__name__}: {str(e)}")
            traceback.print_exc()
            print(f"  Continuing to next dataset...")
            print(f"{'!'*80}\n")
            
            # Clean up GPU memory even on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            continue

print(f"\n{'='*80}")
print(f"BATCH PROCESSING COMPLETE")
print(f"All {len(datasets)} datasets have been processed.")
print(f"{'='*80}")