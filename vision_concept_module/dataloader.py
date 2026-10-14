import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor
import torchvision.transforms as transforms




class CXRDataset(Dataset):
    """
    Alternative implementation using PyTorch transforms for augmentation.
    Use this if you don't want to install Albumentations.
    """
    def __init__(self, df, root_dir='/home/user/cxr_cbm/data/mimic_jpg_cbm', 
                 transform=None, augment=False, augmentation_factor=1):
        """
        Args:
            csv_file: Path to the csv file with image paths and labels
            root_dir: Directory containing the images if paths in CSV are relative
            transform: Optional transform to be applied on a sample
            augment: Whether to apply data augmentation (typically True for training)
            augmentation_factor: How many augmented versions to create per image
        """
       
        self.data_frame = df
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.augmentation_factor = augmentation_factor
        
        # Identify label columns
        self.image_col = 'pa_view_path'
        self.label_cols = [col for col in self.data_frame.columns if '_bin' in col]
        self.label_cols = sorted(self.label_cols)
        self.processor = AutoProcessor.from_pretrained("google/medsiglip-448")
        
        # Define augmentation transforms
        if self.augment:
            self.augmentation_transform = transforms.Compose([
                transforms.RandomAffine(
                    degrees=10,           # Random rotation up to 10 degrees
                    scale=(0.9, 1.1)     # Random zoom 90-110%
    
                ),
                transforms.RandomHorizontalFlip(p=0.3),
            
            ])
        
        # Effective dataset size
        self.original_size = len(self.data_frame)
        self.effective_size = self.original_size * augmentation_factor if augment else self.original_size
        
        print(f"Loaded dataset with {self.original_size} original samples")
        print(f"Effective dataset size with augmentation: {self.effective_size}")
        print(f"Number of disease labels: {len(self.label_cols)}")
        
    def __len__(self):
        return self.effective_size
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Map augmented index back to original index
        original_idx = idx % self.original_size
        
        # Get image path
        img_path = self.data_frame.iloc[original_idx][self.image_col]
        pid, sid, id = img_path.split('/')[-3], img_path.split('/')[-2], img_path.split('/')[-1]
        correct_path = os.path.join(self.root_dir, pid+'_'+sid+'_'+ id+ '.jpg')
        
        # Load image
        image = Image.open(correct_path).convert('RGB')
        
        # Apply augmentation if enabled
        if self.augment:
            image = self.augmentation_transform(image)
        
        # Get all labels
        labels = self.data_frame.iloc[original_idx][self.label_cols].values.astype(np.float32)
        labels = torch.tensor(labels, dtype=torch.float)
        
        return {'image': image, 'labels': labels}


def get_cxr_dataloader(csv_file, root_dir=None, batch_size=32, shuffle=True, num_workers=4):
    """
    Creates a DataLoader for the CXR dataset.
    
    Args:
        csv_file: Path to the csv file with annotations
        root_dir: Directory with all the images
        batch_size: Batch size for the dataloader
        shuffle: Whether to shuffle the dataset
        num_workers: Number of workers for the dataloader
        
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = CXRDataset(csv_file=csv_file, root_dir=root_dir, transform=None)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader

