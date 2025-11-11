import torch
from torch.utils.data import Dataset
from pathlib import Path

class HybridDataset(Dataset):
    def __init__(self, folder_path):
        self.files = list(Path(folder_path).glob("*.pt"))
        if len(self.files) == 0:
            raise ValueError(f"No .pt files found in {folder_path}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location='cpu')
        return (
            data['img_masked'],   # (3, 224, 224)
            data['dct_feat'],     # (1024,)
            torch.tensor(data['label'], dtype=torch.long)
        )