"""
Utility functions for Handwritten Math Solutions.
"""

import editdistance
from typing import List, Optional

import torch
import torchmetrics as tm
from torchmetrics import Metric

class CERMetric(Metric):
    """Character Error Rate metric."""
    
    def __init__(self):
        super().__init__()
        self.add_state("total_cer", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_chars", default=torch.tensor(0), dist_reduce_fx="sum")
        
    def update(self, preds: List[str], targets: List[str]):
        """
        Update metric state.
        
        Args:
            preds: List of predicted strings
            targets: List of target strings
        """
        for pred, target in zip(preds, targets):
            self.total_cer += editdistance.eval(pred, target)
            self.total_chars += len(target)
            
    def compute(self) -> float:
        """Compute the metric."""
        return self.total_cer.float() / self.total_chars

def get_transforms(train: bool = True) -> torchvision.transforms.Compose:
    """
    Get data transforms.
    
    Args:
        train: Whether to get training transforms
        
    Returns:
        Composed transforms
    """
    if train:
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(10),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]) 