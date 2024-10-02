import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

# IMG_PATH = './datasets/image_run2d_dr16q.csv'
SPEC_PATH = './datasets/spec_run2d_dr16q.csv'

IMG_PATH = './datasets/imgs_1000.npy'

IMG_SHAPE = (5, 24, 24)
SPEC_LEN = 3600

class AstroImageSpecDataset(Dataset):
    def __init__(self, img_path=IMG_PATH, spec_path=SPEC_PATH):
        
        # Check file extension
        if img_path.split('.')[-1] == 'csv':
            self.imgs = pd.read_csv(img_path, header=None).values
        elif img_path.split('.')[-1] == 'npy':
            self.imgs = np.load(img_path)
        else:
            raise ValueError('Invalid file extension')
        
        imgs = torch.tensor(self.imgs.reshape(-1, 5, 24*24)).swapaxes(0, 1).squeeze() # (5, N, 24*24)
        # self.imgs = torch.tensor(self.imgs.reshape(-1, IMG_SHAPE)).float().swapaxes(1, 3)
        
        # # Spectrum Loaded as (N, 3600)        
        # self.specs = pd.read_csv(spec_path, header=None).values
        # self.specs = torch.tensor(self.specs.reshape(-1, SPEC_LEN)).float()
        
    def __len__(self):
        return len(self.specs)
    
    def __getitem__(self, index):
        image = self.imgs[index]
        spec = self.specs[index]
        return image, spec

AstroImageSpecDataset()
