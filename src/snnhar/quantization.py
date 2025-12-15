from __future__ import annotations
import copy
import torch
import torch.nn as nn


def model_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def estimate_model_size_bytes(model: nn.Module, bits: int) -> int:
    return model_num_params(model) * bits // 8


def quantize_tensor(t: torch.Tensor, bits: int) -> torch.Tensor:
    if bits >= 32:
        return t
    with torch.no_grad():
        max_val = t.abs().max()
        if max_val == 0:
            return t.clone()
        q_levels = 2 ** (bits - 1) - 1
        scale = max_val / q_levels
        q = torch.round(t / scale).clamp(-q_levels, q_levels)
        return q * scale


def apply_weight_fake_quantization(model: nn.Module, bits: int) -> nn.Module:
    if bits >= 32:
        return copy.deepcopy(model)

    q_model = copy.deepcopy(model)
    with torch.no_grad():
        for _, p in q_model.named_parameters():
            p.data = quantize_tensor(p.data, bits)
    return q_model


def quantize_model_inplace(model: nn.Module, bits: int) -> None:
    if bits >= 32:
        return
    with torch.no_grad():
        for _, p in model.named_parameters():
            p.data = quantize_tensor(p.data, bits)
