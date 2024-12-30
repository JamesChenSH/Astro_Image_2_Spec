import pandas as pd
import torch
from torch.utils.data import Dataset


# IMG_PATH = './datasets/image_run2d_dr16q.csv'
# SPEC_PATH = './datasets/spec_run2d_dr16q.csv'

IMG_PATH = './datasets/imgs_1000.pt'
SPEC_PATH = './datasets/specs_1000.pt'

def preprocess_image(image):    
    return image


def preprocess_spectrum(spectrum):
    # Normalize spectrum
    std, mean = spectrum.std(1, keepdim=True).clip_(0.2), spectrum.mean(1, keepdim=True)
    spectrum = (spectrum - mean) / std
    # Pad spectrum with a single 0 at beginning
    spectrum = torch.nn.functional.pad(spectrum, (1, 0, 0, 0), mode='constant', value=0)
    return spectrum


class AstroImageSpecDataset(Dataset):
    def __init__(self, img_path=IMG_PATH, spec_path=SPEC_PATH, dtype=torch.float32):
        
        # Check file extension
        if img_path.split('.')[-1] == 'csv':
            self.imgs = torch.Tensor(pd.read_csv(img_path, header=None).values)
        elif img_path.split('.')[-1] == 'pt':
            self.imgs = torch.load(img_path, weights_only=False)
        else:
            raise ValueError('Invalid file extension')
        self.imgs = preprocess_image(self.imgs).to(dtype)
        
        # # Spectrum Loaded as (N, 3600)        
        if spec_path.split('.')[-1] == 'csv':
            self.specs = torch.Tensor(pd.read_csv(spec_path, header=None).values)
        elif spec_path.split('.')[-1] == 'pt':
            self.specs = torch.load(spec_path, weights_only=False)
        else:
            raise ValueError('Invalid file extension')
        self.specs = preprocess_spectrum(self.specs).to(dtype)        
        
    def __len__(self):
        return len(self.specs)
    
    def __getitem__(self, index):
        image = self.imgs[index]
        spec = self.specs[index]
        return image, spec


if __name__ == "__main__":
    
    img_path = "./datasets/imgs_10000.pt"
    spec_path = "./datasets/specs_10000.pt"
    
    ds = AstroImageSpecDataset(img_path, spec_path)
    # Save the dataset
    torch.save(ds, './datasets/AstroImg2Spec_ds_10000.pt')