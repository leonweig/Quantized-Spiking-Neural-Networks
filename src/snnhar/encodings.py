from __future__ import annotations
import torch

POP_SIZE_DEFAULT = 4


def rate_encode(x: torch.Tensor, timesteps: int) -> torch.Tensor:
    """
    Poisson/Bernoulli rate encoding.
    x: [B, D] -> spikes: [T, B, D]
    """
    x_norm = torch.sigmoid(x).clamp_(0.0, 1.0)
    x_exp = x_norm.unsqueeze(0).expand(timesteps, -1, -1)
    return torch.bernoulli(x_exp)


def temporal_encode(x: torch.Tensor, timesteps: int) -> torch.Tensor:
    """
    Latency encoding: each feature spikes once; higher value -> earlier.
    Vectorized implementation.
    x: [B, D] -> spikes: [T, B, D]
    """
    B, D = x.shape
    x_norm = torch.sigmoid(x).clamp_(0.0, 1.0)
    spike_times = ((1.0 - x_norm) * (timesteps - 1)).round().long()  # [B, D]

    spikes = torch.zeros(timesteps, B, D, device=x.device, dtype=torch.float32)
    spikes.scatter_(0, spike_times.unsqueeze(0), 1.0)
    return spikes


def population_encode(x: torch.Tensor, timesteps: int, pop_size: int = POP_SIZE_DEFAULT) -> torch.Tensor:
    """
    Gaussian tuning curve -> rate encode population response.
    x: [B, D] -> spikes: [T, B, D*P]
    """
    B, D = x.shape
    device = x.device

    centers = torch.linspace(-2.0, 2.0, pop_size, device=device)  # [P]
    sigma = 1.0

    x_exp = x.unsqueeze(-1)                 # [B, D, 1]
    c = centers.view(1, 1, -1)              # [1, 1, P]
    r = torch.exp(-0.5 * ((x_exp - c) / sigma) ** 2).clamp_(0.0, 1.0)  # [B, D, P]

    r_exp = r.unsqueeze(0).expand(timesteps, -1, -1, -1)  # [T, B, D, P]
    spikes = torch.bernoulli(r_exp).reshape(timesteps, B, D * pop_size).float()
    return spikes


def encode_batch(
    x: torch.Tensor,
    timesteps: int,
    encoding: str,
    pop_size: int = POP_SIZE_DEFAULT,
) -> torch.Tensor:
    enc = encoding.lower()
    if enc == "rate":
        return rate_encode(x, timesteps)
    if enc in ("temporal", "latency"):
        return temporal_encode(x, timesteps)
    if enc == "population":
        return population_encode(x, timesteps, pop_size=pop_size)

    raise ValueError(f"Unknown encoding '{encoding}'. Use rate, temporal/latency, population.")
