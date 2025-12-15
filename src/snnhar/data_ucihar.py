from __future__ import annotations
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from .config import Config


class UCIHARSubset(Dataset):
    def __init__(self, cfg: Config, split: str, scaler: StandardScaler | None = None):
        super().__init__()
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        x_path = os.path.join(cfg.data_root, f"X_{split}.txt")
        y_path = os.path.join(cfg.data_root, f"y_{split}.txt")

        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            raise FileNotFoundError(
                f"Missing dataset files:\n  {x_path}\n  {y_path}\n"
                "Expected:\n"
                "  UCI_HAR_Dataset/X_train.txt\n"
                "  UCI_HAR_Dataset/y_train.txt\n"
                "  UCI_HAR_Dataset/X_test.txt\n"
                "  UCI_HAR_Dataset/y_test.txt\n"
            )

        X = np.loadtxt(x_path)
        y = np.loadtxt(y_path).astype(int)

        y_mapped = np.array([cfg.class_mapping[int(label)] for label in y], dtype=np.int64)

        if scaler is None:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            self.scaler = scaler
        else:
            X = scaler.transform(X)
            self.scaler = scaler

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y_mapped).long()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def get_dataloaders(cfg: Config):
    train_ds = UCIHARSubset(cfg, "train", scaler=None)
    scaler = train_ds.scaler
    test_ds = UCIHARSubset(cfg, "test", scaler=scaler)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    input_dim = train_ds.X.shape[1]
    num_classes = 3
    return train_loader, test_loader, input_dim, num_classes
