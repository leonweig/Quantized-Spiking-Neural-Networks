import torch
from snnhar.models_snn import HARSpikingNet, ConvSpikingNet
from snnhar.models_ann import MLPBaseline


def test_snn_forward_shapes():
    T, B, D = 6, 5, 50
    model = HARSpikingNet(D, 32, 3)
    x = torch.randint(0, 2, (T, B, D)).float()
    logits = model(x)
    assert logits.shape == (B, 3)


def test_conv_snn_forward_shapes():
    T, B, D = 6, 5, 50
    model = ConvSpikingNet(D, 32, 3, conv_channels=4, conv_kernel_size=3)
    x = torch.randint(0, 2, (T, B, D)).float()
    logits = model(x)
    assert logits.shape == (B, 3)


def test_ann_forward_shapes():
    B, D = 7, 50
    model = MLPBaseline(D, 3)
    x = torch.randn(B, D)
    logits = model(x)
    assert logits.shape == (B, 3)
