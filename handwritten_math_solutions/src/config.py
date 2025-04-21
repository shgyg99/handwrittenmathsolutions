"""
Configuration settings for the Handwritten Math Solutions project.
"""

config = {
    "seed": 1234,

    "trainer": {
        "overfit_batches": 0.0,
        "check_val_every_n_epoch": 2,
        "fast_dev_run": False,
        "max_epochs": 100,
        "min_epochs": 1,
        "num_sanity_val_steps": 0,
    },

    "callbacks": {
        "model_checkpoint": {
            "save_top_k": 1,
            "save_weights_only": True,
            "mode": "min",
            "monitor": "val/loss",
            "dirpath": "model_checkpoints",
            "filename": "{epoch}-{val/loss:.2f}-{val/cer:.2f}"
        },
        "early_stopping": {
            "patience": 3,
            "mode": "min",
            "monitor": "val/loss",
            "min_delta": 0.001
        },
        "LearningRateMonitor": {
            "logging_interval": "epoch"
        }
    },

    "data": {
        "batch_size": 16,
        "num_workers": 4,
        "pin_memory": True
    },

    "lit_model": {
        # Optimizer
        "lr": 0.0001,
        "weight_decay": 0.00001,

        # Scheduler
        "milestones": [10],
        "gamma": 0.5,

        # Model
        "d_model": 128,
        "dim_feedforward": 256,
        "nhead": 4,
        "dropout": 0,
        "num_decoder_layers": 3,
        "max_output_len": 200
    },

    "logger": {
        "project": "image-to-latex"
    }
} 