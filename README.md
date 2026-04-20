# From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision–Language Models
## Description
This repository contains the official implementation of the ACL 2026 Findings paper:

**From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision–Language Models**  
Parts of our code are adapted from [V-SEAM](https://github.com/petergit1/V-SEAM) and [Logit Lens](https://github.com/nrimsky/LM-exp). Thanks to the authors for their great work! 

---
We introduce **HONES**, a **H**ead-**O**riented **N**euron **E**xplanation and **S**teering framework for causal interpretability in multi-task Vision–Language Models (VLMs). HONES identifies task-critical FFN neurons by ranking their causal write-in contributions conditioned on task-relevant attention heads, and further supports lightweight neuron-level steering for task improvement. Through large-scale analysis, we find that task-critical neurons show distinct layer preferences across tasks, while shared neurons—particularly those overlapping with **VQA**—play a prominent role in cross-task generalization. Building on these insights, we develop a sparse neuron scaling method that steers key neurons, leading to consistent performance gains on both **LLaVA** and **Qwen** across four diverse multimodal tasks. The figure below demonstrates our framework.

![HONES Framework](https://github.com/petergit1/HONES/blob/main/Image/framework.png)

## Get Start

- [Requirements](#requirements)
- [Critical Head Localization](#critical-head-localization)
- [Head-Conditioned Neuron Attribution](#head-conditioned-neuron-attribution)
- [Neuron Masking Evaluation](#neuron-masking-evaluation)
- [Lightweight Neuron Steering](#lightweight-neuron-steering)

## Requirements

- Python 3.11.10  
- PyTorch 2.2.1+cu121 (CUDA 12.1)

To install all dependencies:

```bash
pip install -r requirements.txt
```

## Critical Head Localization

This step identifies task-critical attention heads through causal head intervention. These heads are used as routing signals for downstream neuron attribution.

For LLAVA:

```bash
python scripts/run_localization.py \
  --model llava \
  --task vqa \
  --config configs/llava_hones.yaml
```

For Qwen2.5-VL:

```bash
python scripts/run_localization.py \
  --model qwen \
  --task vqa \
  --config configs/qwen_hones.yaml
```

## Head-Conditioned Neuron Attribution

This step ranks FFN neurons by their readout-aligned write-in contribution conditioned on the localized task-critical heads.

```bash
python scripts/run_localization.py \
  --model llava \
  --task vqa \
  --config configs/llava_hones.yaml \
  --stage neurons
```

Supported tasks:

```text
vqa
ocr
caption
retrieval
```

## Neuron Masking Evaluation

This step verifies the causal importance of selected neurons by masking them and measuring task performance degradation.

```bash
python scripts/run_masking_eval.py \
  --model llava \
  --task vqa \
  --config configs/llava_hones.yaml
```


## Lightweight Neuron Steering

This step freezes the VLM backbone and learns sparse neuron-wise scaling factors on the identified task-critical neurons.

Run HONES steering:


For LLAVA:

```bash
python scripts/run_steering.py \
  --model llava \
  --task vqa \
  --dataset coco \
  --config configs/llava_hones.yaml
```

For Qwen2.5-VL:

```bash
python scripts/run_steering.py \
  --model qwen \
  --task caption \
  --dataset coco \
  --config configs/qwen_hones.yaml
```

## Cite 
If you find **HONES** useful for your research, please consider citing our work:

```text
@inproceedings{wang-etal-2026-hones,
  title     = {From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision–Language Models},
  author    = {Wang, Qidong  and  Hu, Junjie  and  Jiang, Ming},
  booktitle = {Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
}
