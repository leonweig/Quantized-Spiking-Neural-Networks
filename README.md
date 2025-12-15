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
X_train.txt <br/>
y_train.txt <br/>
X_test.txt <br/>
y_test.txt <br/>

## Install
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


## RUN (everything)

PYTHONPATH=src python -m snnhar.cli --config configs/default.yaml run-all

## RUN (examples)
PYTHONPATH=src python -m snnhar.cli --config configs/default.yaml train-snn --encoding rate --T 20 --epochs 15
PYTHONPATH=src python -m snnhar.cli --config configs/default.yaml sweep-timesteps
PYTHONPATH=src python -m snnhar.cli --config configs/default.yaml sweep-encodings
PYTHONPATH=src python -m snnhar.cli --config configs/default.yaml train-ann --epochs 15


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
