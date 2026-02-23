# SAE Trait Annotation for Organismal Images

Official code release for the ICLR 2026 paper:
[Automatic Image-Level Morphological Trait Annotation for Organismal Images](https://openreview.net/pdf?id=oFRbiaib5Q).

🤗 Dataset: [osunlp/bioscan-traits](https://huggingface.co/datasets/osunlp/bioscan-traits)

## Overview

This repository provides an end-to-end pipeline to:

1. preprocess BIOSCAN-5M into an `ImageFolder` layout,
2. train a Sparse Autoencoder (SAE) on DINOv2 activations,
3. identify species-level prominent latents, and
4. generate natural-language morphological trait annotations using MLLMs (Qwen2.5-VL).

The SAE training/inference stack is adapted from the public [SAEV repository](https://github.com/OSU-NLP-Group/saev) and is included here for convenience.

## Repository Structure

```text
.
|-- preprocess_bioscan.py
|-- create_trait_dataset_mllm_sae.py
|-- create_trait_dataset_mllm.py
|-- saev/                         # SAEV codebase (vendored)
`-- utils/
    |-- create_train_json.py
    `-- convert_trait_wds.py
```

## Environment Setup

This project uses Python 3.11 and `uv`.

```bash
pip install uv
```

Dependencies are installed automatically when running commands via `uv run`.

## Data Preparation

### 1. Download BIOSCAN-5M

```bash
pip install bioscan-dataset
python - <<'PY'
from bioscan_dataset import BIOSCAN5M
_ = BIOSCAN5M("~/Datasets/bioscan-5m", download=True)
PY
```

### 2. Preprocess BIOSCAN-5M into ImageFolder format

`create_trait_dataset_*` scripts expect a `train/` subdirectory under `--data-dir`.

```bash
python preprocess_bioscan.py \
  --csv-file /path/to/bioscan-5m/metadata.csv \
  --image-dir /path/to/bioscan-5m/images \
  --out-dir /path/to/processed_bioscan/train
```

## SAE Pipeline

### 3. Dump DINOv2 activations

```bash
uv run python -m saev activations \
  --vit-family dinov2 \
  --vit-ckpt dinov2_vitb14 \
  --vit-batch-size 1024 \
  --d-vit 768 \
  --n-patches-per-img 256 \
  --vit-layers -2 \
  --dump-to /path/to/activations \
  --n-patches-per-shard 2_4000_000 \
  data:image-folder-dataset \
  --data.root /path/to/processed_bioscan/train
```

### 4. Train SAE

```bash
uv run python -m saev train \
  --data.shard-root /path/to/activations \
  --data.layer -2 \
  --data.patches patches \
  --data.scale-mean False \
  --data.scale-norm False \
  --sae.d-vit 768 \
  --sae.exp-factor 32 \
  --ckpt-path /path/to/sae_ckpt \
  --lr 1e-3 > LOG.txt 2>&1
```

## MLLM Serving

### 5. Start Qwen2.5-VL-72B (vLLM)

```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --host 0.0.0.0 \
  --port <PORT>
```

## Trait Annotation Generation

### 6A. MLLM + SAE

Single-image prompts (`n-img-input=1`):

```bash
uv run python -u create_trait_dataset_mllm_sae.py \
  --data-dir /path/to/processed_bioscan \
  --sae-ckpt-path /path/to/sae_ckpt/sae.pt \
  --thresh 0.9 \
  --trait-thresh 3e-3 \
  --out-dir /path/to/output_mllm_sae \
  --serve-choice qwen_72b \
  --api-url http://0.0.0.0:<PORT>/v1/chat/completions > LOG.txt 2>&1
```

Multi-image prompts (`n-img-input=3`):

```bash
uv run python -u create_trait_dataset_mllm_sae.py \
  --data-dir /path/to/processed_bioscan \
  --sae-ckpt-path /path/to/sae_ckpt/sae.pt \
  --thresh 0.9 \
  --trait-thresh 3e-3 \
  --out-dir /path/to/output_mllm_sae \
  --serve-choice qwen_72b \
  --api-url http://0.0.0.0:<PORT>/v1/chat/completions \
  --n-img-input 3 > LOG.txt 2>&1
```

### 6B. MLLM-only baseline

Single-image prompts:

```bash
uv run python -u create_trait_dataset_mllm.py \
  --data-dir /path/to/processed_bioscan \
  --sae-ckpt-path /path/to/sae_ckpt/sae.pt \
  --thresh 0.9 \
  --trait-thresh 3e-3 \
  --out-dir /path/to/output_mllm \
  --serve-choice qwen_72b \
  --api-url http://0.0.0.0:<PORT>/v1/chat/completions \
  --n-img-input 1 > LOG.txt 2>&1
```

Multi-image prompts:

```bash
uv run python -u create_trait_dataset_mllm.py \
  --data-dir /path/to/processed_bioscan \
  --sae-ckpt-path /path/to/sae_ckpt/sae.pt \
  --thresh 0.9 \
  --trait-thresh 3e-3 \
  --out-dir /path/to/output_mllm \
  --serve-choice qwen_72b \
  --api-url http://0.0.0.0:<PORT>/v1/chat/completions \
  --n-img-input 3 > LOG.txt 2>&1
```

## Expected Outputs

In `--out-dir`, key artifacts include:

- `latent_to_patch_map.json` (MLLM+SAE pipeline),
- `species_latents_prominent/latent_response.jsonl` (model responses),
- per-species annotated patch visualizations under `species_latents_prominent/<species_name>/`.

Use `--debug` and `--n-debug-ex` in generation scripts for small-scale dry runs.

## Fine-tuning Experiments

For downstream classifier training, we build on [BioCLIP](https://github.com/Imageomics/bioclip).  
Preprocessing helpers in this repo:

- `utils/create_train_json.py`: build train JSONs from CSV metadata.
- `utils/convert_trait_wds.py`: convert trait annotations to WebDataset format.

## Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{
pahuja2026automatic,
title={Automatic Image-Level Morphological Trait Annotation for Organismal Images},
author={Vardaan Pahuja and Samuel Stevens and Alyson East and Sydne Record and Yu Su},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=oFRbiaib5Q}
}
```

## Acknowledgments

- [SAEV](https://github.com/OSU-NLP-Group/saev) for sparse autoencoder training infrastructure.
- [BioCLIP](https://github.com/Imageomics/bioclip) for downstream training/evaluation tooling.
