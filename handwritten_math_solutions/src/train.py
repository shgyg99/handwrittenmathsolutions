"""
Training module for Handwritten Math Solutions.
"""

import os
from pathlib import Path

import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor
)
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.config import config
from src.data.dataset import MathDataset
from src.model.math_model import MathTransformer

class MathLightningModule(LightningModule):
    """Lightning module for training the math recognition model."""
    
    def __init__(self, model: MathTransformer, config: dict):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        formulas = batch['formula']
        
        outputs = self.model(images, formulas[:, :-1])
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), formulas[:, 1:].reshape(-1))
        
        self.log('train/loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        formulas = batch['formula']
        
        outputs = self.model(images, formulas[:, :-1])
        loss = self.criterion(outputs.view(-1, outputs.size(-1)), formulas[:, 1:].reshape(-1))
        
        self.log('val/loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config['lit_model']['lr'],
            weight_decay=config['lit_model']['weight_decay']
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['lit_model']['gamma'],
            patience=2,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val/loss'
        }

def train():
    """Main training function."""
    # Set random seed
    torch.manual_seed(config['seed'])
    
    # Initialize model
    model = MathTransformer(
        d_model=config['lit_model']['d_model'],
        nhead=config['lit_model']['nhead'],
        num_decoder_layers=config['lit_model']['num_decoder_layers'],
        dim_feedforward=config['lit_model']['dim_feedforward'],
        dropout=config['lit_model']['dropout'],
        max_output_len=config['lit_model']['max_output_len']
    )
    
    # Initialize lightning module
    lit_model = MathLightningModule(model, config)
    
    # Initialize callbacks
    callbacks = [
        ModelCheckpoint(**config['callbacks']['model_checkpoint']),
        EarlyStopping(**config['callbacks']['early_stopping']),
        LearningRateMonitor(**config['callbacks']['LearningRateMonitor'])
    ]
    
    # Initialize logger
    logger = WandbLogger(**config['logger'])
    
    # Initialize trainer
    trainer = Trainer(
        callbacks=callbacks,
        logger=logger,
        **config['trainer']
    )
    
    # Train model
    trainer.fit(lit_model)

if __name__ == '__main__':
    train() 