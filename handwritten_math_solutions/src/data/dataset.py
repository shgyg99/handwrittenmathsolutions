"""
Dataset module for Handwritten Math Solutions.
"""

import os
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as TT

class MathDataset(Dataset):
    """Custom dataset for handwritten math solutions."""
    
    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        transform: Optional[TT.Compose] = None,
        max_len: int = 200
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Directory with all the images
            csv_file: Path to the CSV file with annotations
            transform: Optional transform to be applied on a sample
            max_len: Maximum length of the output sequence
        """
        self.root_dir = Path(root_dir)
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        self.max_len = max_len
        
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = self.annotations.iloc[idx, 0]
        img_path = self.root_dir / img_name
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        formula = self.annotations.iloc[idx, 1]
        
        sample = {
            'image': image,
            'formula': formula
        }
        
        return sample 