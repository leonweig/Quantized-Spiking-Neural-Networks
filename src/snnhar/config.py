from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml
import torch


@dataclass
class Config:
    # Data
    data_root: str = "./UCI_HAR_Dataset"

    # Training
    batch_size: int = 64
    num_workers: int = 0
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    seed: int = 42

    # SNN
    timesteps_main: int = 20
    num_epochs_main: int = 15
    encoding_main: str = "rate"
    lambda_spike: float = 1e-4

    # QAT (simple finetune)
    run_qat: bool = True
    qat_bits: int = 8
    qat_epochs: int = 5
    qat_lr_scale: float = 0.1

    # Sweeps
    run_timestep_sweep: bool = True
    timesteps_sweep: tuple = (5, 10, 20, 40)
    num_epochs_sweep: int = 8

    run_encoding_sweep: bool = True
    encodings_to_compare: tuple = ("rate", "temporal", "population")
    num_epochs_encoding: int = 8

    # ConvSNN
    run_conv_snn: bool = True
    num_epochs_conv: int = 10
    conv_channels: int = 16
    conv_kernel_size: int = 5

    # ANN
    run_ann_baseline: bool = True
    num_epochs_ann: int = 15

    # Class mapping
    class_mapping: Optional[Dict[int, int]] = None

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}

    cfg = Config(**d)
    if cfg.class_mapping is None:
        cfg.class_mapping = {
            1: 0, 2: 0, 3: 0,  # walking-related
            4: 1, 5: 1,        # sitting/standing
            6: 2,              # lying
        }

    # YAML lists -> tuples where expected
    if isinstance(cfg.timesteps_sweep, list):
        cfg.timesteps_sweep = tuple(cfg.timesteps_sweep)
    if isinstance(cfg.encodings_to_compare, list):
        cfg.encodings_to_compare = tuple(cfg.encodings_to_compare)

    return cfg
