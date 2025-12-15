from __future__ import annotations
import torch
import torch.nn as nn

try:
    from norse.torch.module.lif import LIFCell
except ImportError as e:
    raise ImportError("Install norse: pip install norse") from e


class HARSpikingNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.lif = LIFCell()
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_spikes: torch.Tensor, return_spike_sum: bool = False):
        """
        x_spikes: [T, B, D]
        """
        T, B, _ = x_spikes.shape
        lif_state = None

        z = torch.zeros(B, self.hidden_dim, device=x_spikes.device, dtype=x_spikes.dtype)
        spike_sum = torch.zeros(B, device=x_spikes.device) if return_spike_sum else None

        for t in range(T):
            h = self.fc_in(x_spikes[t])
            z, lif_state = self.lif(h, lif_state)
            if return_spike_sum:
                spike_sum += z.abs().sum(dim=1)

        logits = self.fc_out(z.to(self.fc_out.weight.dtype))
        return (logits, spike_sum) if return_spike_sum else logits


class ConvSpikingNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        conv_channels: int = 16,
        conv_kernel_size: int = 5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=conv_channels,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
        )
        self.fc_in = nn.Linear(conv_channels * input_dim, hidden_dim)
        self.lif = LIFCell()
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_spikes: torch.Tensor, return_spike_sum: bool = False):
        T, B, D = x_spikes.shape
        lif_state = None

        z = torch.zeros(B, self.hidden_dim, device=x_spikes.device, dtype=x_spikes.dtype)
        spike_sum = torch.zeros(B, device=x_spikes.device) if return_spike_sum else None

        for t in range(T):
            x_t = x_spikes[t].unsqueeze(1)       # [B,1,D]
            conv_out = self.conv(x_t)            # [B,C,D]
            feat = conv_out.reshape(B, -1)       # [B,C*D]
            h = self.fc_in(feat)
            z, lif_state = self.lif(h, lif_state)
            if return_spike_sum:
                spike_sum += z.abs().sum(dim=1)

        logits = self.fc_out(z.to(self.fc_out.weight.dtype))
        return (logits, spike_sum) if return_spike_sum else logits
