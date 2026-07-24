# SSP
Learning by Survival: Semantic Survivor Principle for Cross-Modal Retrieval with Partial Labels

Official implementation of the paper **"Learning by Survival: Semantic Survivor Principle for Cross-Modal Retrieval with Partial Labels"**.

## Abstract

Partial-label cross-modal retrieval aims to align semantically related instances across heterogeneous modalities, where each sample is associated with a candidate label set instead of a uniquely specified ground-truth label. Although this weakly supervised paradigm substantially reduces annotation cost, it introduces severe semantic ambiguity. Existing methods generally address this challenge by independently estimating the reliability of candidate semantics. However, the objective of partial-label learning is to identify the unique ground-truth semantic from multiple ambiguous candidates, making semantic selection inherently a relative competitive decision problem rather than an independent confidence estimation problem. To this end, we propose the Semantic Survivor Principle (SSP), a new learning principle that models partial-label cross-modal retrieval as a competitive collaborative semantic selection process. Specifically, candidate semantics compete within each modality, allowing reliable survivors to progressively emerge through temporal semantic competition, while cross-modal collaboration further reinforces survivors that maintain consistent semantic evidence across heterogeneous modalities. This competitive collaborative mechanism enables reliable semantics to dominate optimization while continuously suppressing noisy candidates throughout training. Guided by SSP, we develop a unified framework consisting of two complementary components. Survivor Competitive Learning (SCL) explicitly models the temporal evolution of candidate semantics via recursive state updates, progressively reinforcing semantically stable candidates according to their historical stability while suppressing unstable ones. Building upon the stabilized survivor semantics, Cross-modal Survivor Alignment (CSA) constructs a shared probabilistic semantic representation to collaboratively align semantic distributions across modalities, thereby promoting consistent semantic evolution in the shared embedding space and learning robust, discriminative cross-modal representations. Extensive experiments on four public benchmarks demonstrate the effectiveness and superiority of the proposed framework. 

## Quick Start

### 1. Environment

- Python ≥ 3.8  
- PyTorch ≥ 1.10.0  
- CUDA ≥ 11.3  

Install dependencies:

```bash
pip install numpy pandas scikit-learn tqdm matplotlib
```

## 2. Run Experiments

### Single Task

```bash
python train.py --dataset wiki --partial_length 3 --lr 1e-4
```

### Batch Task

Run preconfigured batch experiments for each dataset:

wiki:
```bash
bash run_wiki.sh
```

NUS-WIDE:
```bash
bash run_nus-wide.sh
```

INRIA-Websearch:
```bash
bash run_INRIA-Websearch.sh
```

XMediaNet:
```bash
bash run_xmedianet.sh
```
