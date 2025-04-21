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
from torch import nn
from .config import config
from .model.math_model import MathTransformer
from tokenizers import Tokenizer
from .data.dataloader import data_loader
from typing import List, Optional, Set
from torchmetrics import Metric
from torch import Tensor
import editdistance


class CharacterErrorRate(Metric):
    def __init__(self, ignore_indices: Set[int], *args):
        # Initialize the Metric class and set the ignore indices for certain tokens
        super().__init__(*args)
        self.ignore_indices = ignore_indices

        # Track the total errors and the number of samples processed
        self.add_state("error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.error: Tensor
        self.total: Tensor

    def update(self, preds, targets):
        N = preds.shape[0]

        for i in range(N):
            # Filter out ignored indices from both prediction and target
            pred = [token for token in preds[i].tolist() if token not in self.ignore_indices]
            target = [token for token in targets[i].tolist() if token not in self.ignore_indices]

            # Compute the edit distance (Levenshtein distance) between prediction and target
            distance = editdistance.distance(pred, target)

            # Update error count based on the relative length of the prediction and target
            if max(len(pred), len(target)) > 0:
                self.error += distance / max(len(pred), len(target))

        # Increment total processed samples
        self.total += N

    def compute(self) -> Tensor:
        # Return the average character error rate
        return self.error / self.total
    

class MathLightningModule(LightningModule):
    def __init__(self,
                 d_model: int,
                 dim_feedforward: int,
                 nhead: int,
                 dropout: float,
                 num_decoder_layers: int,
                 max_output_len: int,
                 lr: float = 0.001,
                 weight_decay: float = 0.0001,
                 milestones: List[int] = [5],
                 gamma: float = 0.1,
                 save_path: str = 'path_to_save'):  # Default save path for model
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.milestones = milestones
        self.gamma = gamma
        
        self.save_hyperparameters()
        self.tokenizer = Tokenizer.from_file(f"handwritten_math_solutions/src/latex_tokenizer.json")  # Load tokenizer
        self.model = MathTransformer(  # Initialize ResNet-Transformer model
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            nhead=nhead,
            dropout=dropout,
            num_decoder_layers=num_decoder_layers,
            max_output_len=max_output_len,
            sos_index=self.tokenizer.get_vocab()["<s>"],  # Start-of-sequence token
            eos_index=self.tokenizer.get_vocab()["</s>"],  # End-of-sequence token
            pad_index=self.tokenizer.get_vocab()["<pad>"],  # Padding token
            num_classes=self.tokenizer.get_vocab_size(),  # Number of classes
        )
        # Cross-Entropy loss ignoring padding
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.get_vocab()["<pad>"])
        self.val_cer = CharacterErrorRate({self.tokenizer.get_vocab()["<pad>"], self.tokenizer.get_vocab()["<s>"], self.tokenizer.get_vocab()["</s>"]})  # CER for validation
        self.test_cer = CharacterErrorRate({self.tokenizer.get_vocab()["<pad>"], self.tokenizer.get_vocab()["<s>"], self.tokenizer.get_vocab()["</s>"]})  # CER for testing

    def training_step(self, batch):
        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])  # Model output excluding last target token
        loss = self.loss_fn(logits, targets[:, 1:])  # Compute loss excluding first target token
        self.log("train/loss", loss)  # Log training loss
        return loss
    
    def validation_step(self, batch):

        imgs, targets = batch
        logits = self.model(imgs, targets[:, :-1])  # Model output
        loss = self.loss_fn(logits, targets[:, 1:])  # Compute loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)  # Log validation loss

        preds = self.model.predict(imgs)  # Get predictions
        val_cer = self.val_cer(preds, targets)  # Compute CER for validation
        self.log("val/cer", val_cer)

    def test_step(self, batch):
        imgs, targets = batch
        preds = self.model.predict(imgs)  # Get predictions
        test_cer = self.test_cer(preds, targets)  # Compute CER for test
        self.log("test/cer", test_cer)  # Log test CER
        self.test_step_outputs.append(preds)  # Store predictions
        return preds
    

    def on_test_epoch_end(self):
        with open(f"{self.path}/test_predictions.txt", "w") as f:  # Save predictions to file
            for preds in self.test_step_outputs:
                for pred in preds:
                    decoded = [self.tokenizer.id_to_token(j) for j in pred.tolist() if j != 3]  # Decode predictions
                    decoded.append("\n")
                    f.write(" ".join(decoded))  # Write to file

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # AdamW optimizer
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.gamma)  # Learning rate scheduler
        return [optimizer], [scheduler]

def train():
    """Main training function."""
    # Set random seed
    torch.manual_seed(config['seed'])
    tokenizer = Tokenizer.from_file(f"handwritten_math_solutions/src/latex_tokenizer.json") 
    # Initialize model
    model = MathTransformer(
        d_model=config['lit_model']['d_model'],
        nhead=config['lit_model']['nhead'],
        num_decoder_layers=config['lit_model']['num_decoder_layers'],
        dim_feedforward=config['lit_model']['dim_feedforward'],
        dropout=config['lit_model']['dropout'],
        max_output_len=config['lit_model']['max_output_len'],
        sos_index=tokenizer.get_vocab()["<s>"],  # Start-of-sequence token
        eos_index=tokenizer.get_vocab()["</s>"],  # End-of-sequence token
        pad_index=tokenizer.get_vocab()["<pad>"],  # Padding token
        num_classes=tokenizer.get_vocab_size(), 
    )
    
    # Initialize lightning module
    lit_model = MathLightningModule(**config['lit_model'])
    
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
    train_loader, valid_loader, test_loader = data_loader()
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

if __name__ == '__main__':
    train() 