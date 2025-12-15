# SNN HAR Pipeline (UCI HAR)

End-to-end pipeline for Human Activity Recognition:
- Spiking Neural Networks (Norse LIFCell + surrogate gradients)
- Encodings: rate, temporal/latency, population
- Spike-count regularization
- Quantization experiments (FP32/FP16/INT8/INT4) + INT8 finetuning
- Baselines: MLP ANN, ConvSNN variant
- Sweeps: timesteps and encoding comparisons
- Outputs: plots + checkpoints + metrics.json

## Dataset
Expected files (561-feature text format):
UCI_HAR_Dataset/
X_train.txt
y_train.txt
X_test.txt
y_test.txt

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

## RUN (everything)

python -m snnhar.cli run-all --config configs/default.yaml

## RUN (examples)
python -m snnhar.cli train-snn --config configs/default.yaml --encoding rate --T 20 --epochs 15
python -m snnhar.cli sweep-timesteps --config configs/default.yaml
python -m snnhar.cli sweep-encodings --config configs/default.yaml
python -m snnhar.cli train-ann --config configs/default.yaml --epochs 15

## Outputs
Saved under:
- outputs/<run_id>/checkpoints/
- outputs/<run_id>/plots/
- outputs/<run_id>/metrics.json

## Label mapping
UCI HAR 6 classes -> 3 macro-classes:
- walking-related -> 0
- sitting/standing -> 1
- lying -> 2