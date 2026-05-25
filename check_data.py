import torch
from pre_processing import EllipticDataset, IBMAMLDataset_HiSmall, IBMAMLDataset_LiSmall, IBMAMLDataset_HiMedium, IBMAMLDataset_LiMedium, AMLSimDataset

datasets = [
    ('Elliptic',         lambda: EllipticDataset(root='Datasets/Elliptic_dataset')[0]),
    ('IBM_AML_HiSmall',  lambda: IBMAMLDataset_HiSmall(root='Datasets/IBM_AML_dataset/HiSmall')[0]),
    ('IBM_AML_LiSmall',  lambda: IBMAMLDataset_LiSmall(root='Datasets/IBM_AML_dataset/LiSmall')[0]),
    ('IBM_AML_HiMedium', lambda: IBMAMLDataset_HiMedium(root='Datasets/IBM_AML_dataset/HiMedium')[0]),
    ('IBM_AML_LiMedium', lambda: IBMAMLDataset_LiMedium(root='Datasets/IBM_AML_dataset/LiMedium')[0]),
    ('AMLSim',           lambda: AMLSimDataset(root='Datasets/AMLSim_dataset')[0]),
]

for name, loader in datasets:
    data = loader()
    col = data.edge_index[1]
    sorted_ok = bool(torch.all(col[1:] >= col[:-1]).item())
    status = 'OK (column-sorted)' if sorted_ok else 'STALE (needs re-processing)'
    print(f'{name:20s}: {status}')
