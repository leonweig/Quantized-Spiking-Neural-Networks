from __future__ import annotations
import argparse
import os

from .config import load_config
from .experiments import (
    make_run_dir,
    run_main_snn,
    run_timestep_sweep,
    run_encoding_sweep,
    run_conv_snn,
    run_ann,
    save_metrics,
)
from .plots import (
    plot_quantization_results,
    plot_timestep_sweep,
    plot_encoding_results,
    plot_snn_vs_ann,
)


def cmd_train_snn(args):
    cfg = load_config(args.config)
    out_dir = make_run_dir(args.outputs)
    res = run_main_snn(cfg, out_dir, encoding=args.encoding, T=args.T, epochs=args.epochs)

    plot_dir = os.path.join(out_dir, "plots")
    plot_quantization_results(res["snn_results"], plot_dir, "quantization_snn_main")
    save_metrics(out_dir, cfg, {"main_snn": res["snn_results"]})
    print(out_dir)


def cmd_sweep_timesteps(args):
    cfg = load_config(args.config)
    out_dir = make_run_dir(args.outputs)
    res = run_main_snn(cfg, out_dir)  # to get loaders/dims consistent

    sweep = run_timestep_sweep(cfg, res["train_loader"], res["test_loader"], res["input_dim"], res["num_classes"], res["encoding"])
    plot_timestep_sweep(sweep, os.path.join(out_dir, "plots"), "timesteps_sweep")
    save_metrics(out_dir, cfg, {"timestep_sweep": sweep})
    print(out_dir)


def cmd_sweep_encodings(args):
    cfg = load_config(args.config)
    out_dir = make_run_dir(args.outputs)
    res = run_main_snn(cfg, out_dir)  # for loaders/dims

    enc = run_encoding_sweep(cfg, res["train_loader"], res["test_loader"], res["input_dim"], res["num_classes"])
    plot_encoding_results(enc, os.path.join(out_dir, "plots"), "encoding_sweep")
    save_metrics(out_dir, cfg, {"encoding_sweep": enc})
    print(out_dir)


def cmd_train_ann(args):
    cfg = load_config(args.config)
    out_dir = make_run_dir(args.outputs)
    res = run_main_snn(cfg, out_dir)  # to share loaders/dims
    ann_results = run_ann(cfg, res["train_loader"], res["test_loader"], res["input_dim"], res["num_classes"])
    save_metrics(out_dir, cfg, {"ann": ann_results})
    print(out_dir)


def cmd_run_all(args):
    cfg = load_config(args.config)
    out_dir = make_run_dir(args.outputs)

    main = run_main_snn(cfg, out_dir)
    plot_dir = os.path.join(out_dir, "plots")

    plot_quantization_results(main["snn_results"], plot_dir, "quantization_snn_main")

    payload = {"main_snn": main["snn_results"]}

    if cfg.run_timestep_sweep:
        sweep = run_timestep_sweep(cfg, main["train_loader"], main["test_loader"], main["input_dim"], main["num_classes"], main["encoding"])
        plot_timestep_sweep(sweep, plot_dir, "timesteps_sweep")
        payload["timestep_sweep"] = sweep

    if cfg.run_encoding_sweep:
        enc = run_encoding_sweep(cfg, main["train_loader"], main["test_loader"], main["input_dim"], main["num_classes"])
        plot_encoding_results(enc, plot_dir, "encoding_sweep")
        payload["encoding_sweep"] = enc

    if cfg.run_conv_snn:
        conv = run_conv_snn(cfg, main["train_loader"], main["test_loader"], main["input_dim"], main["num_classes"], main["encoding"], main["T"])
        payload["conv_snn"] = conv

    ann_results = None
    if cfg.run_ann_baseline:
        ann_results = run_ann(cfg, main["train_loader"], main["test_loader"], main["input_dim"], main["num_classes"])
        payload["ann"] = ann_results
        plot_snn_vs_ann(main["snn_results"], ann_results, plot_dir, "snn_vs_ann")

    save_metrics(out_dir, cfg, payload)
    print(out_dir)


def build_parser():
    p = argparse.ArgumentParser(prog="snnhar", description="SNN HAR pipeline CLI")
    p.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--outputs", default="outputs", help="Base output directory")

    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("train-snn", help="Train main SNN + quantization")
    s1.add_argument("--encoding", default=None, help="Override encoding (rate/temporal/population)")
    s1.add_argument("--T", type=int, default=None, help="Override timesteps")
    s1.add_argument("--epochs", type=int, default=None, help="Override epochs")
    s1.set_defaults(func=cmd_train_snn)

    s2 = sub.add_parser("sweep-timesteps", help="Run timestep sweep")
    s2.set_defaults(func=cmd_sweep_timesteps)

    s3 = sub.add_parser("sweep-encodings", help="Run encoding sweep")
    s3.set_defaults(func=cmd_sweep_encodings)

    s4 = sub.add_parser("train-ann", help="Train ANN baseline")
    s4.set_defaults(func=cmd_train_ann)

    s5 = sub.add_parser("run-all", help="Run full pipeline")
    s5.set_defaults(func=cmd_run_all)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
