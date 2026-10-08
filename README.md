# Single-Pass Uncertainty Heads for Claim-Level Hallucination Detection in Persian Medical Language Models

**Code, datasets, trained uncertainty heads, and reproduction resources for the paper:**

> **Single-Pass Uncertainty Heads for Claim-Level Hallucination Detection in Persian Medical Language Models**
> Mehrdad Ghassabi, Pedram Rostami, Hamidreza Baradaran Kashani, Sadra Hakim, Audrina Ebrahimi

[![arXiv](https://img.shields.io/badge/arXiv-2610.03482-b31b1b.svg)](https://arxiv.org/abs/2610.03482)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-gaokerena-yellow)](https://huggingface.co/gaokerena)

## Paper

**[arXiv:2610.03482 — Single-Pass Uncertainty Heads for Claim-Level Hallucination Detection in Persian Medical Language Models](https://arxiv.org/abs/2610.03482)**

The paper studies whether a lightweight uncertainty head can identify hallucinated claims from a **single forward pass** of a Persian medical language model.

We adapt the **LLM Uncertainty Head (LUH)** framework to two Aya-Expanse-8B-based Persian medical models:

* **Gaokerena-V**
* **Gaokerena-R**

The uncertainty heads operate on frozen backbone signals—attention history and token probabilities—and predict a hallucination score at the claim level. No retrieval or repeated sampling is required during inference.

> **Important limitation:** the held-out test sets contain only 50 responses per backbone, and the claim labels were generated automatically rather than by human annotators. The reported results should therefore be regarded as an initial study rather than a definitive benchmark.

## What this repository provides

This repository contains the resources needed to reproduce the experiments:

* training and evaluation code;
* claim-level Persian hallucination datasets;
* trained uncertainty heads for Gaokerena-V and Gaokerena-R;
* documentation and reproduction notes;
* corrections needed when adapting the public LUH implementation to the models used here.

## Main results

On the held-out claim-level test sets:

| Metric          | Gaokerena-V | Gaokerena-R |
| --------------- | ----------: | ----------: |
| Accuracy        |       76.24 |       62.96 |
| Precision       |       44.85 |       29.52 |
| Recall          |       58.17 |       80.84 |
| F1              |       50.65 |       43.24 |
| ROC-AUC         |   **78.52** |   **78.10** |
| PR-AUC          |   **48.20** |   **46.52** |
| Random PR-AUC   |       20.95 |       17.46 |
| PR-AUC / random |   **2.30×** |   **2.66×** |
| Test claims     |       1,951 |       1,687 |

Accuracy is not sufficient for interpreting these results because the classes are imbalanced. A classifier that never flags a claim already obtains 79.05% accuracy for Gaokerena-V and 82.54% for Gaokerena-R. PR-AUC and ROC-AUC are therefore more informative.

## Released resources

All released artifacts are available through the [`gaokerena`](https://huggingface.co/gaokerena) Hugging Face organization.

### Trained uncertainty heads

| Backbone    | Trained head                                                                      |
| ----------- | --------------------------------------------------------------------------------- |
| Gaokerena-V | [gaokerena/gaokerena-V-uhead](https://huggingface.co/gaokerena/gaokerena-V-uhead) |
| Gaokerena-R | [gaokerena/gaokerena-R-uhead](https://huggingface.co/gaokerena/gaokerena-R-uhead) |

### Claim-level hallucination datasets

The datasets are constructed directly in **Persian**, without translating the generated responses into another language.

| Dataset | Backbone    | Link                                                                                   |
| ------- | ----------- | -------------------------------------------------------------------------------------- |
| LUH-V   | Gaokerena-V | [gaokerena/LUH_Gaokerena_V](https://huggingface.co/datasets/gaokerena/LUH_Gaokerena_V) |
| LUH-R   | Gaokerena-R | [gaokerena/LUH_Gaokerena_R](https://huggingface.co/datasets/gaokerena/LUH_Gaokerena_R) |

### Backbone models

The two Gaokerena models are distributed as LoRA adapters built on [Cohere Aya-Expanse-8B](https://huggingface.co/CohereLabs/aya-expanse-8b).

| Model       | Link                                                                        |
| ----------- | --------------------------------------------------------------------------- |
| Gaokerena-V | [gaokerena/gaokerena-v1.0](https://huggingface.co/gaokerena/gaokerena-v1.0) |
| Gaokerena-R | [gaokerena/gaokerena-r1.0](https://huggingface.co/gaokerena/gaokerena-r1.0) |

## Dataset

Each backbone was evaluated on the same **1,600 Persian medical questions**, using greedy decoding.

| Statistic             |  LUH-V |  LUH-R |
| --------------------- | -----: | -----: |
| Responses             |  1,600 |  1,600 |
| Extracted claims      | 60,434 | 54,209 |
| Supported claims      | 48,991 | 43,305 |
| Hallucinated claims   | 10,557 |  9,528 |
| Undecided claims      |    886 |  1,376 |
| Hallucination rate    | 17.73% | 18.03% |
| Supervised tokens     |  61.4% |  54.9% |
| Positive-class weight |   4.64 |   4.53 |

The response-level split is:

* **1,458** training responses
* **92** validation responses
* **50** test responses

The test set is kept separate from checkpoint selection and model selection.

### Dataset construction

1. A deterministic set of 1,600 Persian medical questions was selected.
2. Both Gaokerena-V and Gaokerena-R answered the same questions in the same order using greedy decoding.
3. Atomic, verifiable claims were extracted from the Persian responses.
4. Each claim was labeled as **supported**, **hallucinated**, or **undecided** using DeepSeek-V4-Flash.
5. Supported and hallucinated claims were aligned with their token spans under the Aya-Expanse tokenizer.
6. Undecided claims remain in the released datasets but are excluded from the supervised loss.

## Uncertainty-head architecture

For each frozen backbone, the uncertainty head uses:

1. attention history from all backbone layers, with history size 3;
2. the top-4 token probabilities at each position.

These signals are concatenated and passed through a lightweight Transformer encoder:

| Setting               | Value             |
| --------------------- | ----------------- |
| Hidden dimension      | 768               |
| Transformer layers    | 2                 |
| Attention heads       | 8                 |
| Dropout               | 0.1               |
| Backbone              | Frozen            |
| Backbone precision    | fp16              |
| Learning rate         | 1e-4              |
| Warmup ratio          | 0.05              |
| Weight decay          | 0.1               |
| Batch size            | 8                 |
| Gradient accumulation | 1                 |
| Maximum epochs        | 10                |
| Early stopping        | 3 epochs          |
| Checkpoint metric     | Validation PR-AUC |

The head produces token-level hallucination scores, which are pooled over each extracted claim to obtain a **claim-level hallucination score**.

No hyperparameter search was performed. Training used a single CUDA GPU with approximately 40 GB of memory.

## Reproducing the experiments

### 1. Load the datasets and trained heads

```python
from datasets import load_dataset
from huggingface_hub import snapshot_download

luh_v = load_dataset("gaokerena/LUH_Gaokerena_V")
luh_r = load_dataset("gaokerena/LUH_Gaokerena_R")

head_v = snapshot_download("gaokerena/gaokerena-V-uhead")
head_r = snapshot_download("gaokerena/gaokerena-R-uhead")
```

### 2. Merge the Gaokerena LoRA adapter

The Gaokerena models are distributed as LoRA adapters. The LUH training pipeline used here operates on a standalone backbone, so the adapter must first be merged into Aya-Expanse-8B.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_id = "CohereLabs/aya-expanse-8b"
adapter_id = "gaokerena/gaokerena-v1.0"  # or gaokerena/gaokerena-r1.0

base = AutoModelForCausalLM.from_pretrained(
    base_id,
    torch_dtype=torch.float16
)

model = PeftModel.from_pretrained(base, adapter_id)
model = model.merge_and_unload()

model.save_pretrained("merged-gaokerena-v")
AutoTokenizer.from_pretrained(base_id).save_pretrained(
    "merged-gaokerena-v"
)
```

Before training, verify tokenizer compatibility and confirm that chat-template tokenization reproduces the stored `input_ids`.

### 3. Train a head

Run the training code for either Gaokerena-V or Gaokerena-R.

The **92-response `eval` split** should be used for validation and checkpoint selection. The **50-response test split must remain untouched** until final evaluation.

### 4. Evaluate

Evaluation is performed separately from training.

Load the selected checkpoint and evaluate it on the held-out test split. The released heads can also be evaluated directly using the corresponding Hugging Face checkpoints.

The repository contains the implementation and supporting documentation under [`src/`](https://github.com/Mehrdadghassabi/gaokerena-luh/tree/main/src) and [`doc/`](https://github.com/Mehrdadghassabi/gaokerena-luh/tree/main/doc).

## Response-variability motivation experiment

Before training the uncertainty heads, the study examined output variability on the **168-question September 2023 Iranian Basic Medical Sciences Entrance Examination (IBMSEE)** using five chain-of-thought runs per model.

| Metric                    | Gaokerena-V | Gaokerena-R | Aya-Expanse | Med-Gemma |
| ------------------------- | ----------: | ----------: | ----------: | --------: |
| Ensemble accuracy (%)     |       29.76 |       38.69 |       35.71 |     37.50 |
| Single option in all runs |          14 |          37 |          36 |       168 |
| All four options observed |          10 |           4 |           6 |         0 |
| Entropy (bits)            |        1.11 |        0.80 |        0.84 |         0 |
| Parameters                |          8B |          8B |          8B |        4B |

Here, the ensemble answer is the option selected by at least three of the five runs. Entropy is the mean per-question Shannon entropy over valid runs.

This experiment motivates the need for uncertainty estimation without requiring multiple generations at inference time.

## Implementation corrections

Two issues in the public LUH implementation were corrected before training.

### Chat-template boundary

Recent Transformers versions can return a mapping from `apply_chat_template` rather than a plain token sequence. Treating the mapping length as the token count can shift the reply boundary and incorrectly place claim masks on prompt tokens.

The implementation here explicitly extracts the `input_ids` sequence before determining the reply boundary.

### Precision handling

The original model-loading path did not always resolve the configured PyTorch dtype correctly. As a result, a run configured for fp16 could still load the 8B backbone in fp32.

The implementation here explicitly resolves the requested PyTorch dtype before model loading.

These corrections affect preprocessing and model loading; they do **not** change the uncertainty-head architecture.

## Limitations

This project is an initial study of claim-level uncertainty estimation for Persian medical language models.

The main limitations are:

* the held-out test split contains only **50 responses per backbone**;
* claim labels are **automatically generated**, not manually annotated;
* the uncertainty heads are trained separately for specific frozen backbones;
* the study evaluates two closely related Persian medical models rather than a broad set of architectures;
* the results should not be interpreted as establishing clinical reliability.

The models and uncertainty heads are research artifacts and are **not intended to provide medical advice**.

## Citation

If you use this code, datasets, or trained heads, please cite the paper:

```bibtex
@article{ghassabi2026singlepass,
  title   = {Single-Pass Uncertainty Heads for Claim-Level Hallucination Detection in Persian Medical Language Models},
  author  = {Ghassabi, Mehrdad and
             Rostami, Pedram and
             Baradaran Kashani, Hamidreza and
             Hakim, Sadra and
             Ebrahimi, Audrina},
  journal = {arXiv preprint arXiv:2610.03482},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.03482}
}
```

Please also cite the underlying LUH and Gaokerena works:

* **LUH:** Shelmanov et al., *A Head to Predict and a Head to Question: Pre-trained Uncertainty Quantification Heads for Hallucination Detection in LLM Outputs*, EMNLP 2025. [arXiv:2505.08200](https://arxiv.org/abs/2505.08200)
* **Gaokerena-V:** Ghassabi et al., *Leveraging Online Data to Enhance Medical Knowledge in a Small Persian Language Model*. [arXiv:2505.16000](https://arxiv.org/abs/2505.16000)
* **Gaokerena-R:** Ghassabi et al., *Enhancing Reasoning Skills in Small Persian Medical Language Models Can Outperform Large-Scale Data Training*. [arXiv:2510.20059](https://arxiv.org/abs/2510.20059)

## Authors

**Mehrdad Ghassabi**, **Pedram Rostami**, **Hamidreza Baradaran Kashani**, **Sadra Hakim**, **Audrina Ebrahimi**


