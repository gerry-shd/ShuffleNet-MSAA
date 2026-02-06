# ShuffleNet-MSAA

Lightweight sonar-image classifier that couples a ShuffleNet-style backbone with Multi-Scale Adaptive Attention (MSAA) blocks and Dual-Weighted Focal Loss (DW-FL). The training pipeline focuses on class-imbalance robustness and aggressive data augmentation so the model can be deployed on resource-constrained hardware.

## Features
- MSAA blocks mix channel attention and spatial gating across three kernel sizes.
- Multi-Scale Extractors and stochastic depth keep the backbone efficient.
- Optional mixup/cutmix, RandAugment, EMA tracking, and cosine LR schedule built in.
- Dataset utilities discover class folders or honor predefined train/val/test directories.
- Evaluation helpers write JSON metrics, confusion matrices (CSV + PNG), and label maps.

## Requirements
```bash
python -m venv .venv
.\.venv\Scripts\activate    # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```
The project targets Python 3.10+ with PyTorch 2.x and CUDA (when available). CPU-only mode is also supported.

## Dataset Layout
Default discovery mode expects the following structure:
```
DATA_ROOT/
  class_a/ img001.jpg ...
  class_b/ ...
```
Alternatively, set `data.split_mode: folders` and provide explicit `train/`, `val/`, `test/` directories inside `config.yaml->data.split_folders`.

## Configuration
All hyper-parameters live in `config.yaml`.
- `paths.data_root`: Root folder of the dataset.
- `paths.output_dir`: Training artifacts directory.
- `model`: Architecture name, width multiplier, drop-path rate, and pretrained source (see `config.py`).
- `data`: Image size, workers, augmentation knobs, and split ratios.
- `training`: Epochs, seed, loss function (ce|focal|dwfl), optimizer, mixup schedule, EMA, etc.

Track experiment configs by copying the file into `outputs/<run>/config_used.yaml` (handled automatically by `train.py`).

## Training
```
python train.py --config config.yaml
```
Key command-line switches:
- `--config`: Alternate YAML file.
- `--no_amp`: Force-disable mixed precision even if `training.amp` is true.

The script creates `outputs/<run>/` with checkpoints (`best.pt`, `last.pt`), logs (`train_log.csv`), `label_map.json`, and `splits.json` (if the dataset was auto-split).

## Evaluation
```
python eval.py --checkpoint outputs/<run>/best.pt --config config.yaml \
               --data_root <DATA_ROOT> --out_dir outputs/<run>
```
Results: `eval/test_metrics.json`, `eval/confusion_matrix.(png|csv)`, and console summaries (Top-1, Macro-F1, Weighted-F1).

## Prediction
```
python predict.py --checkpoint outputs/<run>/best.pt --image path/to/image.jpg --topk 5
```
Prints the best classes and probabilities for a single image (uses the test transforms defined in the config).

## Project Structure
```
config.py       # pretrained helpers
config.yaml     # default hyper-parameters
dataset.py      # discovery, splitting, augmentations
eval.py         # offline evaluation entry point
losses.py       # CE/FL/DW-FL implementations
metrics.py      # report + confusion-matrix utilities
models.py       # ShuffleNet-MSAA definition
predict.py      # single-image inference script
train.py        # full training loop
utils.py        # assorted helpers (EMA, logging, etc.)
```

## Tips
1. Set `model.pretrained` to one of the options listed in `config.yaml` to warm-start from ImageNet weights.
2. When using auto splits, delete `outputs/<run>/splits.json` if you need a fresh stratified shuffle.
3. Monitor `train_log.csv` with tools like TensorBoard or Pandas for quick diagnostics.
4. DW-FL requires class counts, so ensure the training split contains at least one image per class.
