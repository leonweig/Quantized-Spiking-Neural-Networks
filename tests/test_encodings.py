import torch
from snnhar.encodings import rate_encode, temporal_encode, population_encode


def test_rate_encode_shape_and_binary():
    x = torch.randn(8, 561)
    s = rate_encode(x, timesteps=10)
    assert s.shape == (10, 8, 561)
    assert torch.all((s == 0) | (s == 1))


def test_temporal_encode_shape_and_one_spike_per_feature():
    x = torch.randn(4, 12)
    s = temporal_encode(x, timesteps=7)
    assert s.shape == (7, 4, 12)
    # each feature should spike exactly once per sample
    counts = s.sum(dim=0)  # [B, D]
    assert torch.all(counts == 1)


def test_population_encode_shape():
    x = torch.randn(3, 20)
    s = population_encode(x, timesteps=5, pop_size=4)
    assert s.shape == (5, 3, 80)
