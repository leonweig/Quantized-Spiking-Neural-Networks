from __future__ import annotations
import os
import json
import time
import copy
from dataclasses import asdict
import torch
import torch.nn as nn

from .config import Config
from .data_ucihar import get_dataloaders
from .encodings import POP_SIZE_DEFAULT
from .models_snn import HARSpikingNet, ConvSpikingNet
from .models_ann import MLPBaseline
from .train import set_seed, train_one_epoch_snn, eval_snn, train_one_epoch_ann, eval_ann
from .quantization import (
    estimate_model_size_bytes,
    apply_weight_fake_quantization,
    quantize_model_inplace,
)


def make_run_dir(base: str = "outputs") -> str:
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base, run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir


def train_qat_from_pretrained(
    base_model: nn.Module,
    train_loader,
    test_loader,
    bits: int,
    device: str,
    timesteps: int,
    encoding: str,
    epochs: int,
    base_lr: float,
    lr_scale: float = 0.1,
    lambda_spike: float = 0.0,
):
    qat_model = copy.deepcopy(base_model).to(device)
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=base_lr * lr_scale)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        qat_model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            from .encodings import encode_batch
            x_spikes = encode_batch(x, timesteps, encoding).to(device)

            optimizer.zero_grad(set_to_none=True)
            if lambda_spike > 0.0:
                logits, spike_sum = qat_model(x_spikes, return_spike_sum=True)
                loss = criterion(logits, y) + lambda_spike * spike_sum.mean()
            else:
                logits = qat_model(x_spikes, return_spike_sum=False)
                loss = criterion(logits, y)

            loss.backward()
            optimizer.step()
            quantize_model_inplace(qat_model, bits)

        _vl, _va, _ = eval_snn(qat_model, test_loader, device, timesteps, encoding)

    val_loss, val_acc, latency = eval_snn(qat_model, test_loader, device, timesteps, encoding)
    return qat_model, {
        "mode": f"int{bits}_qat",
        "bits": bits,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "latency_s": latency,
        "model_size_bytes": estimate_model_size_bytes(qat_model, bits),
    }


def run_main_snn(cfg: Config, out_dir: str, encoding: str | None = None, T: int | None = None, epochs: int | None = None):
    set_seed(cfg.seed)
    train_loader, test_loader, input_dim, num_classes = get_dataloaders(cfg)

    encoding = encoding or cfg.encoding_main
    T = T or cfg.timesteps_main
    epochs = epochs or cfg.num_epochs_main

    main_input_dim = input_dim * POP_SIZE_DEFAULT if encoding == "population" else input_dim

    model = HARSpikingNet(main_input_dim, cfg.hidden_dim, num_classes).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for _ in range(epochs):
        train_one_epoch_snn(model, train_loader, optimizer, cfg.device, T, encoding, lambda_spike=cfg.lambda_spike)

    ckpt_path = os.path.join(out_dir, "checkpoints", "har_snn_fp32.pt")
    torch.save(model.state_dict(), ckpt_path)

    results = []

    # FP32
    val_loss, val_acc, latency = eval_snn(model, test_loader, cfg.device, T, encoding)
    results.append({
        "mode": "fp32", "bits": 32,
        "val_loss": val_loss, "val_acc": val_acc, "latency_s": latency,
        "model_size_bytes": estimate_model_size_bytes(model, 32),
    })

    # FP16
    fp16_model = copy.deepcopy(model).half()
    # Eval with float loss compute
    from .encodings import encode_batch
    @torch.no_grad()
    def eval_fp16(m):
        m.eval()
        crit = nn.CrossEntropyLoss()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        t0 = time.perf_counter()
        for x, y in test_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            x_spikes = encode_batch(x, T, encoding).to(cfg.device).half()
            logits = m(x_spikes, return_spike_sum=False)
            loss = crit(logits.float(), y)
            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)
        return total_loss / total_samples, total_correct / total_samples, time.perf_counter() - t0

    vl, va, lat = eval_fp16(fp16_model)
    results.append({
        "mode": "fp16", "bits": 16,
        "val_loss": vl, "val_acc": va, "latency_s": lat,
        "model_size_bytes": estimate_model_size_bytes(fp16_model, 16),
    })

    # INT8 / INT4 PTQ
    for name, bits in [("int8", 8), ("int4", 4)]:
        q_model = apply_weight_fake_quantization(model, bits).to(cfg.device)
        vl, va, lat = eval_snn(q_model, test_loader, cfg.device, T, encoding)
        results.append({
            "mode": name, "bits": bits,
            "val_loss": vl, "val_acc": va, "latency_s": lat,
            "model_size_bytes": estimate_model_size_bytes(q_model, bits),
        })

    # QAT (simple)
    if cfg.run_qat:
        _, qat_res = train_qat_from_pretrained(
            base_model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            bits=cfg.qat_bits,
            device=cfg.device,
            timesteps=T,
            encoding=encoding,
            epochs=cfg.qat_epochs,
            base_lr=cfg.learning_rate,
            lr_scale=cfg.qat_lr_scale,
            lambda_spike=cfg.lambda_spike,
        )
        results.append(qat_res)

    return {
        "snn_model": model,
        "snn_results": results,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "encoding": encoding,
        "T": T,
    }


def run_timestep_sweep(cfg: Config, train_loader, test_loader, input_dim: int, num_classes: int, encoding: str):
    sweep = []
    base_input_dim = input_dim * POP_SIZE_DEFAULT if encoding == "population" else input_dim

    for T in cfg.timesteps_sweep:
        set_seed(cfg.seed)
        model = HARSpikingNet(base_input_dim, cfg.hidden_dim, num_classes).to(cfg.device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        for _ in range(cfg.num_epochs_sweep):
            train_one_epoch_snn(model, train_loader, opt, cfg.device, T, encoding, lambda_spike=cfg.lambda_spike)

        vl, va, lat = eval_snn(model, test_loader, cfg.device, T, encoding)
        q_model = apply_weight_fake_quantization(model, 8).to(cfg.device)
        ql, qa, qlat = eval_snn(q_model, test_loader, cfg.device, T, encoding)

        sweep.append({
            "T": T,
            "fp32_acc": va, "fp32_loss": vl, "fp32_latency": lat,
            "fp32_size_bytes": estimate_model_size_bytes(model, 32),
            "int8_acc": qa, "int8_loss": ql, "int8_latency": qlat,
            "int8_size_bytes": estimate_model_size_bytes(q_model, 8),
        })

    return sweep


def run_encoding_sweep(cfg: Config, train_loader, test_loader, input_dim: int, num_classes: int):
    from .encodings import POP_SIZE_DEFAULT
    enc_results = []
    for enc in cfg.encodings_to_compare:
        set_seed(cfg.seed)
        enc_input_dim = input_dim * POP_SIZE_DEFAULT if enc == "population" else input_dim
        model = HARSpikingNet(enc_input_dim, cfg.hidden_dim, num_classes).to(cfg.device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

        for _ in range(cfg.num_epochs_encoding):
            train_one_epoch_snn(model, train_loader, opt, cfg.device, cfg.timesteps_main, enc, lambda_spike=cfg.lambda_spike)

        vl, va, lat = eval_snn(model, test_loader, cfg.device, cfg.timesteps_main, enc)
        enc_results.append({"encoding": enc, "val_loss": vl, "val_acc": va, "latency_s": lat})

    return enc_results


def run_conv_snn(cfg: Config, train_loader, test_loader, input_dim: int, num_classes: int, encoding: str, T: int):
    conv_input_dim = input_dim * POP_SIZE_DEFAULT if encoding == "population" else input_dim
    set_seed(cfg.seed)
    model = ConvSpikingNet(
        conv_input_dim,
        cfg.hidden_dim,
        num_classes,
        conv_channels=cfg.conv_channels,
        conv_kernel_size=cfg.conv_kernel_size,
    ).to(cfg.device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    for _ in range(cfg.num_epochs_conv):
        train_one_epoch_snn(model, train_loader, opt, cfg.device, T, encoding, lambda_spike=cfg.lambda_spike)

    vl, va, lat = eval_snn(model, test_loader, cfg.device, T, encoding)
    return {
        "val_loss": vl,
        "val_acc": va,
        "latency_s": lat,
        "size_bytes": estimate_model_size_bytes(model, 32),
    }


def run_ann(cfg: Config, train_loader, test_loader, input_dim: int, num_classes: int):
    set_seed(cfg.seed)
    model = MLPBaseline(input_dim, num_classes).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for _ in range(cfg.num_epochs_ann):
        train_one_epoch_ann(model, train_loader, opt, cfg.device)

    vl, va, lat = eval_ann(model, test_loader, cfg.device)
    results = [{
        "mode": "ann_fp32",
        "bits": 32,
        "val_loss": vl, "val_acc": va, "latency_s": lat,
        "model_size_bytes": estimate_model_size_bytes(model, 32),
    }]

    q_model = apply_weight_fake_quantization(model, 8).to(cfg.device)
    ql, qa, qlat = eval_ann(q_model, test_loader, cfg.device)
    results.append({
        "mode": "ann_int8",
        "bits": 8,
        "val_loss": ql, "val_acc": qa, "latency_s": qlat,
        "model_size_bytes": estimate_model_size_bytes(q_model, 8),
    })

    return results


def save_metrics(out_dir: str, cfg: Config, payload: dict):
    data = {
        "config": asdict(cfg),
        "results": payload,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
