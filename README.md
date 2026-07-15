# SSP
Learning by Survival: Semantic Survivor Principle for Cross-Modal Retrieval with Partial Labels

Official implementation of the paper **"Learning by Survival: Semantic Survivor Principle for Cross-Modal Retrieval with Partial Labels"**.

## Abstract

Partial-label cross-modal retrieval aims to align semantically related instances across heterogeneous modalities, where each sample is associated with a candidate label set rather than a uniquely specified ground-truth label. This weakly supervised manner substantially reduces annotation cost but inevitably introduces semantic ambiguity during training. Existing approaches typically attempt to alleviate this ambiguity by estimating the reliability of candidate labels during training. However, such methods evaluate candidate semantics in an isolated and static manner, without considering their evolving interactions throughout the optimization process, which may lead to unstable supervision and degraded representation learning. To address this limitation, we propose the Semantic Survivor Principle (SSP), which conceptualizes partial-label cross-modal learning as a temporally evolving competitive process under candidate-set constraints. Rather than evaluating candidate semantics in isolation, SSP models them as competing units whose contributions to parameter updates are dynamically regulated during training. Semantics that demonstrate sustained temporal stability and cross-modal consistency are progressively reinforced, whereas noisy candidates are gradually suppressed, thereby enabling reliable semantics to dominate the representation learning process. Building upon this principle, we develop a unified framework comprising two cooperative components: Survivor Competitive Learning (SCL) employs a first-order Markov recursive update mechanism to explicitly capture the temporal evolution of candidate semantic weights and to stabilize intra-modal semantic distributions during optimization; Cross-modal Survivor Alignment (CSA) further constructs a joint probabilistic representation by aggregating the stabilized semantic distributions from each modality, thereby promoting stable and collaborative cross-modal optimization within the shared embedding space. Extensive experiments on four public benchmarks demonstrate the superiority of the proposed method.

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
python train.py --dataset wiki --partial_ratio 0.1 --lr 1e-4
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
