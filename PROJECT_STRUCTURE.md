# SpatialThinker Project Structure

This document provides an overview of each file and directory in this repository, along with information about training data sources.

---

## Training Data Source

The training data for SpatialThinker comes from **STVQA-7K**, a high-quality spatial visual question answering dataset created as part of this research project.

- **Hugging Face Dataset**: [`hunarbatra/STVQA-7K`](https://huggingface.co/datasets/hunarbatra/STVQA-7K)
- **Description**: Contains ~7,000 spatial VQA samples with images, questions, and ground truth answers with bounding boxes and scene graph annotations
- **Format**: Parquet files with train/validation splits
- **Fields**:
  - `problem` - The question/prompt text
  - `answer` / `answer_option_text` - Ground truth answers
  - `images` - Image data

---

## Root Directory Files

| File | Description |
|------|-------------|
| `README.md` | Main project documentation with installation, training, and evaluation instructions |
| `setup.py` | Python package setup script for installing the `verl` package |
| `pyproject.toml` | Python project configuration with build system and ruff linting settings |
| `requirements.txt` | Python dependencies required to run the project |
| `.gitignore` | Git ignore patterns |

---

## Directories

### `evaluation/`
Contains evaluation scripts for benchmarking models on spatial reasoning tasks.

| File | Description |
|------|-------------|
| `evals.py` | Main evaluation script supporting 18+ benchmarks (CV-Bench, BLINK, SpatialBench, etc.) with support for HuggingFace, OpenAI, and Anthropic models |
| `templates.py` | Prompt templates including `SPATIAL_THINKER_TEMPLATE` for structured scene graph reasoning |

---

### `scripts/`
Training and utility scripts.

| File | Description |
|------|-------------|
| `config.yaml` | Default training configuration (hyperparameters, data settings, worker configs) |
| `spatialthinker_3b_grpo.sh` | Train SpatialThinker-3B with dense spatial rewards + GRPO |
| `spatialthinker_7b_grpo.sh` | Train SpatialThinker-7B with dense spatial rewards + GRPO |
| `qwen_2_5_3b_stvqa_vanilla_grpo.sh` | Baseline 3B model with vanilla GRPO (no spatial rewards) |
| `qwen_2_5_7b_stvqa_vanilla_grpo.sh` | Baseline 7B model with vanilla GRPO (no spatial rewards) |
| `model_merger.py` | Merge training checkpoints to Hugging Face format |
| `runtime_env.yaml` | Ray runtime environment configuration |

#### `scripts/extras/`
Additional experimental training scripts for other datasets (GeoQA, CLEVR, etc.).

---

### `verl/`
Core RL training framework based on veRL/EasyR1.

#### `verl/trainer/`
Training loop and algorithms.

| File | Description |
|------|-------------|
| `main.py` | Entry point for training (`python -m verl.trainer.main`) |
| `ray_trainer.py` | Ray-based distributed trainer implementation |
| `core_algos.py` | RL algorithms including GRPO, REINFORCE, and advantage estimation |
| `config.py` | Training configuration dataclasses |
| `metrics.py` | Training metrics and logging utilities |

#### `verl/workers/`
Distributed worker implementations.

| Subdirectory | Description |
|--------------|-------------|
| `actor/` | Actor model workers for policy updates |
| `critic/` | Critic model workers (for algorithms that use critics) |
| `reward/` | Reward computation workers |
| `rollout/` | Rollout generation workers using vLLM |
| `sharding_manager/` | FSDP sharding utilities |
| `fsdp_workers.py` | Fully Sharded Data Parallel worker base classes |
| `config.py` | Worker configuration dataclasses |

#### `verl/utils/`
Utility functions and helper modules.

| File | Description |
|------|-------------|
| `dataset.py` | Dataset loading and preprocessing (supports Hugging Face and local parquet) |
| `tokenizer.py` | Tokenizer utilities |
| `torch_functional.py` | PyTorch helper functions |
| `torch_dtypes.py` | Data type utilities |
| `model_utils.py` | Model loading utilities |
| `seqlen_balancing.py` | Sequence length balancing for efficient batching |
| `fsdp_utils.py` | FSDP helper functions |
| `flops_counter.py` | FLOPS counting utilities |
| `py_functional.py` | Python helper functions |
| `ulysses.py` | Ulysses sequence parallelism utilities |

#### `verl/utils/reward_score/`
Reward functions for RL training.

| File | Description |
|------|-------------|
| `spatial_sgg.py` | **Core spatial reward function** - Dense rewards based on scene graph matching (object detection, bounding box IoU, relationship matching) |
| `r1v.py` | R1V reward function (format + accuracy rewards) |
| `r1v_scene.py` | R1V reward with scene understanding |
| `math.py` | Math problem reward function |

#### `verl/utils/logger/`
Logging integrations (WandB, SwanLab, console).

#### `verl/utils/checkpoint/`
Checkpoint saving and loading utilities.

#### `verl/models/`
Model-specific code.

| File | Description |
|------|-------------|
| `monkey_patch.py` | Model monkey patches for compatibility |
| `transformers/` | Transformer model utilities (Qwen2-VL RoPE indexing, etc.) |

#### `verl/single_controller/`
Single-controller distributed training utilities.

| Subdirectory | Description |
|--------------|-------------|
| `base/` | Base controller classes |
| `ray/` | Ray-specific controller implementations |

#### `verl/protocol.py`
Data protocol definitions for inter-worker communication.

---

### `assets/`
Project assets (images for README, etc.).

---

## Key Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  1. Data: STVQA-7K dataset (HuggingFace)                   │
│  2. Base Model: Qwen2.5-VL-3B/7B-Instruct                  │
│  3. Algorithm: GRPO (Group Relative Policy Optimization)   │
│  4. Reward: Dense spatial rewards (spatial_sgg.py)         │
│     - Object detection matching                             │
│     - Bounding box IoU/CIoU                                │
│     - Relationship triplet matching                         │
│     - Format + accuracy rewards                            │
│  5. Output: SpatialThinker-3B/7B checkpoints               │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

**Train SpatialThinker:**
```bash
bash scripts/spatialthinker_7b_grpo.sh
```

**Evaluate:**
```bash
python3 evaluation/evals.py --dataset spatialbench --model_path OX-PIXL/SpatialThinker-7B
```

**Merge checkpoint:**
```bash
python3 scripts/model_merger.py --local_dir path_to_checkpoint
```
