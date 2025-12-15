from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .encodings import encode_batch


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch_snn(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    timesteps: int,
    encoding: str,
    lambda_spike: float = 0.0,
):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        x_spikes = encode_batch(x, timesteps, encoding).to(device)

        optimizer.zero_grad(set_to_none=True)

        if lambda_spike > 0.0:
            logits, spike_sum = model(x_spikes, return_spike_sum=True)
            loss = criterion(logits, y) + lambda_spike * spike_sum.mean()
        else:
            logits = model(x_spikes, return_spike_sum=False)
            loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def eval_snn(model: nn.Module, dataloader: DataLoader, device: str, timesteps: int, encoding: str):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    t0 = time.perf_counter()
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        x_spikes = encode_batch(x, timesteps, encoding).to(device)
        logits = model(x_spikes, return_spike_sum=False)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    latency = time.perf_counter() - t0
    return total_loss / total_samples, total_correct / total_samples, latency


def train_one_epoch_ann(model: nn.Module, dataloader: DataLoader, optimizer, device: str):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def eval_ann(model: nn.Module, dataloader: DataLoader, device: str):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    t0 = time.perf_counter()
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += y.size(0)

    latency = time.perf_counter() - t0
    return total_loss / total_samples, total_correct / total_samples, latency
