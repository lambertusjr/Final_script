from dependencies import *
from utilities import *

def process_raw_data():

    from pre_processing import EllipticDataset, IBMAMLDataset_HiSmall, IBMAMLDataset_LiSmall, IBMAMLDataset_HiMedium, IBMAMLDataset_LiMedium, AMLSimDataset
    
    data = EllipticDataset(root='Datasets/Elliptic_dataset')[0]
    data = IBMAMLDataset_HiSmall(root='Datasets/IBM_AML_dataset/HiSmall')[0]
    data = IBMAMLDataset_LiSmall(root='Datasets/IBM_AML_dataset/LiSmall')[0]
    data = IBMAMLDataset_HiMedium(root='Datasets/IBM_AML_dataset/HiMedium')[0]
    data = IBMAMLDataset_LiMedium(root='Datasets/IBM_AML_dataset/LiMedium')[0]
    data = AMLSimDataset(root='Datasets/AMLSim_dataset')[0]
    del data
    print("All pre_processing complete.")

process_raw_data()