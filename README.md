# PIXL-R1: Visual Spatial Reasoning Model with GRPO

## Features

- Supported models
  - Llama3/Qwen2/Qwen2.5 language models
  - Qwen2/Qwen2.5-VL vision language models
  - DeepSeek-R1 distill models

- Supported algorithms
  - GRPO
  - Reinforce++
  - Remax
  - RLOO

- Supported datasets
  - Any text, vision-text dataset in a [specific format](#custom-dataset)

- Supported tricks
  - Padding-free training
  - Resuming from checkpoint
  - Wandb & SwanLab tracking

## Requirements

### Software Requirements

- Python 3.9+
- transformers>=4.49.0
- flash-attn>=2.4.3
- vllm>=0.7.3 (0.8.0 is recommended)

### Installation

```bash
pip install -e .
```

### GRPO Training

To Train with Geo3k
```bash
bash scripts/qwen2_5_vl_7b_geo3k.sh
```
To train with CLEVR-70k-Counting (Counting problems)

```bash
bash scripts/qwen2_5_vl_7b_clevr.sh
```
To train with GeoQA-8k 

```bash
bash scripts/qwen2_5_vl_7b_geoqa8k.sh
```

### Merge Checkpoint in Hugging Face Format

```bash
python3 scripts/model_merger.py --local_dir path_to_your_last_actor_checkpoint
```

## Custom Dataset

Please refer to the example datasets to prepare your own dataset.

- Text dataset: https://huggingface.co/datasets/hiyouga/math12k
- Vision-text dataset: https://huggingface.co/datasets/hiyouga/geometry3k (Supports multi-image dataset)
