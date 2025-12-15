from __future__ import annotations
import os
import matplotlib.pyplot as plt


def plot_quantization_results(results, out_dir: str, fname_prefix: str):
    modes = [r["mode"] for r in results]
    accs = [r["val_acc"] for r in results]
    sizes_kb = [r["model_size_bytes"] / 1024 for r in results]

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.bar(modes, accs)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Accuracy")
    plt.xlabel("Precision mode")
    plt.title("Accuracy vs Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fname_prefix}_accuracy.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(modes, sizes_kb)
    plt.ylabel("Model size [KB]")
    plt.xlabel("Precision mode")
    plt.title("Model Size vs Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fname_prefix}_size.png"), dpi=150)
    plt.close()


def plot_timestep_sweep(sweep_results, out_dir: str, fname_prefix: str):
    Ts = [r["T"] for r in sweep_results]
    fp32_acc = [r["fp32_acc"] for r in sweep_results]
    int8_acc = [r["int8_acc"] for r in sweep_results]

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(Ts, fp32_acc, marker="o", label="FP32")
    plt.plot(Ts, int8_acc, marker="s", label="INT8 PTQ")
    plt.ylim(0.0, 1.05)
    plt.xlabel("Timesteps T")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Timesteps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fname_prefix}_accuracy_vs_T.png"), dpi=150)
    plt.close()


def plot_encoding_results(enc_results, out_dir: str, fname_prefix: str):
    encodings = [r["encoding"] for r in enc_results]
    accs = [r["val_acc"] for r in enc_results]

    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(encodings, accs)
    plt.ylim(0.0, 1.05)
    plt.xlabel("Encoding")
    plt.ylabel("Accuracy")
    plt.title("Encoding Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fname_prefix}_accuracy.png"), dpi=150)
    plt.close()


def plot_snn_vs_ann(snn_results, ann_results, out_dir: str, fname_prefix: str):
    snn_fp32 = next(r for r in snn_results if r["mode"] == "fp32")
    snn_int8 = next(r for r in snn_results if r["mode"] == "int8")
    ann_fp32 = next(r for r in ann_results if r["mode"] == "ann_fp32")
    ann_int8 = next(r for r in ann_results if r["mode"] == "ann_int8")

    labels = ["SNN FP32", "SNN INT8", "ANN FP32", "ANN INT8"]
    accs = [snn_fp32["val_acc"], snn_int8["val_acc"], ann_fp32["val_acc"], ann_int8["val_acc"]]
    sizes_kb = [
        snn_fp32["model_size_bytes"] / 1024,
        snn_int8["model_size_bytes"] / 1024,
        ann_fp32["model_size_bytes"] / 1024,
        ann_int8["model_size_bytes"] / 1024,
    ]

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.bar(labels, accs)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("SNN vs ANN Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fname_prefix}_accuracy.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.bar(labels, sizes_kb)
    plt.ylabel("Model size [KB]")
    plt.title("SNN vs ANN Model Size")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{fname_prefix}_size.png"), dpi=150)
    plt.close()
