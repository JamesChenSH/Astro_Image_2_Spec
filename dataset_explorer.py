import torch
from torch.utils.data import Dataset
from utils.dataset_builder import AstroImageSpecDataset

if __name__ == "__main__":
    # Load the dataset
    dataset = torch.load("./datasets/AstroImg2Spec_ds_1000.pt", weights_only=False)
    
    img, spec = dataset[0]
    
    # Print max and min values of the spectrum
    print(f"Max value of the spectrum: {spec.max()}")
    print(f"Min value of the spectrum: {spec.min()}")

    # Print the shape of the image and spectrum
    print(f"Shape of the image: {img.shape}")
    print(f"Shape of the spectrum: {spec.shape}")
    
    