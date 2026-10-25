from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class NPZFloodDataset(Dataset):
    """One .npz per event/tile. Keys: sar_pre, sar_post [1,H,W];
    ms_pre, ms_post [3,H,W]; mask [1,H,W]. Arrays must be normalized.
    The mask is binary flood change, not generic water extent.
    """
    def __init__(self,folder):
        self.files=sorted(Path(folder).glob('*.npz'))
        if not self.files: raise FileNotFoundError(f'No .npz tiles in {folder}')
    def __len__(self): return len(self.files)
    def __getitem__(self,index):
        with np.load(self.files[index]) as f:
            sample={key:torch.from_numpy(np.asarray(f[key],dtype=np.float32).copy()) for key in ('sar_pre','sar_post','ms_pre','ms_post','mask')}
        hw=sample['mask'].shape[-2:]
        for k,v in sample.items():
            if v.ndim!=3 or v.shape[-2:]!=hw: raise ValueError(f'{self.files[index]} bad shape for {k}: {v.shape}')
        if not torch.all((sample['mask']==0)|(sample['mask']==1)):
            raise ValueError('mask must contain only zero and one')
        return sample
