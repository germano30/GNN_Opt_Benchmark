# Which is the best Optimizer for your Graph Neural Network?

While research has focused on improving Graph Neural Network (GNN) architectures, little work has evaluated the impact of optimization algorithms on GNN training. This paper presents a comprehensive evaluation study and practical guidelines for choosing the optimization methods for GNNs problems. We evaluate the performance of SGD, AdamW, Shampoo, SOAP, and Muon across link prediction, node classification and graph classification tasks in both homogeneous and heterogeneous graphs. The evaluated architectures were GCN, GAT, GIN, RGCN, and RGAT.
## Overview

| Dimension | Coverage |
|---|---|
| **Optimizers** | AdamW, SGD, Shampoo, SOAP, Muon |
| **Architectures** | GCN, GAT, GIN (+ BatchNorm / LayerNorm / no norm), RGCN, RGAT |
| **Tasks** | Node classification, Link prediction, Graph classification |
| **Datasets** | Cora, ogbn-proteins, ogbl-collab, ogbg-ppa, WordNet18RR |

## Requirements

**Step 1 — PyTorch (via conda, CUDA 12.1):**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

**Step 2 — Sparse extensions (pre-compiled wheels):**
```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

**Step 3 — Remaining dependencies:**
```bash
pip install -r requirements.txt
```

> A CUDA-capable GPU is required. The benchmark will raise an error if no GPU is available.

## Project Structure

```
GNN_Muon/
├── benchmark_v2.py        # Main experiment runner
├── prepare_datasets.py    # Dataset download and preprocessing
├── benchmark.yaml         # Default experiment configuration
├── requirements.txt
├── models/
│   ├── gcn.py             # Graph Convolutional Network
│   ├── gat.py             # Graph Attention Network
│   ├── gin.py             # Graph Isomorphism Network (with norm variants)
│   ├── rgcn.py            # Relational GCN (heterogeneous graphs)
│   └── rgat.py            # Relational GAT (heterogeneous graphs)
└── utils/
    ├── data.py            # Dataset catalog and loaders
    ├── training.py        # Training and evaluation loops
    ├── optimizers.py      # Optimizer factory
    ├── predictors.py      # Task-specific prediction heads
    ├── wrappers.py        # Node embedding wrapper for feature-less graphs
    └── soap.py            # SOAP optimizer implementation
```

## Preparing Datasets

Download and preprocess all datasets before running experiments:

```bash
python prepare_datasets.py --dataset all
```

Or prepare a single dataset:

```bash
python prepare_datasets.py --dataset Cora
python prepare_datasets.py --dataset ogbl-collab
python prepare_datasets.py --dataset ogbn-proteins
python prepare_datasets.py --dataset ogbg-ppa
python prepare_datasets.py --dataset WordNet18RR
```

Processed files are saved to `dataset/processed_<name>.pt`.

## Running Experiments

### Via YAML config (recommended)

```bash
python benchmark_v2.py --config benchmark.yaml
```

The YAML file controls all hyperparameters, target datasets, models, and optimizers. Key fields:

```yaml
experiment:
  runs: 5
  epochs: 200
  batch_size: 1024
  patience: 30        # early stopping patience

hyperparameters:
  lr: 0.001
  weight_decay: 0.01
  hidden_channels: 256
  num_layers: 3
  dropout: 0.5
  grad_accum_steps: 1
  num_neighbors: [15, 10, 5]

targets:
  datasets: [Cora, ogbl-collab, ogbn-proteins, ogbg-ppa, WordNet18RR]
  models: [GCN, GAT, GIN_layer, RGCN, RGAT]
  optimizers: [AdamW, SGD, Muon, Shampoo, SOAP]
```

### Via command line

```bash
# Run a single combination
python benchmark_v2.py --dataset Cora --model GCN --optimizer AdamW --epochs 200 --runs 5

# Run all optimizers on a given dataset/model
python benchmark_v2.py --dataset ogbl-collab --model GIN_layer --optimizer all --runs 5
```

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | — | Path to YAML config (overrides other args) |
| `--dataset` | `Cora` | Dataset name |
| `--model` | `GCN` | Model architecture |
| `--optimizer` | `AdamW` | Optimizer name, or `all` |
| `--epochs` | `10` | Number of training epochs |
| `--lr` | `0.01` | Learning rate |
| `--batch_size` | `1024` | Mini-batch size |
| `--runs` | `1` | Number of independent runs (mean ± std reported) |

## Datasets

| Dataset | Task | Graph type | Metric |
|---|---|---|---|
| Cora | Node classification | Homogeneous | Accuracy |
| ogbn-proteins | Node classification | Homogeneous | ROC-AUC |
| ogbl-collab | Link prediction | Homogeneous | Hits@50 |
| ogbg-ppa | Graph classification | Homogeneous | Accuracy |
| WordNet18RR | Link prediction | Heterogeneous (KG) | MRR |

## Optimizers

| Optimizer | Description |
|---|---|
| **AdamW** | Adaptive first-order baseline with decoupled weight decay |
| **SGD** | Stochastic gradient descent with momentum |
| **Shampoo** | Approximate second-order with Kronecker-factored preconditioner |
| **SOAP** | Preconditioned Adam with Shampoo-like eigenbasis updates |
| **Muon** | Newton-Schulz orthogonalisation applied to 2-D weight gradients; 1-D parameters and embeddings fall back to AdamW |

> **Note on Muon:** embedding parameters (any parameter whose name contains `embed`) are automatically excluded from the Newton-Schulz update and handled by the internal AdamW sub-group. This is required for knowledge-graph datasets (WordNet18RR) and any model using a `NodeEmbeddingWrapper`.

## Output

Each experiment run creates a timestamped results directory:

```
results_YYYYMMDD_HHMMSS/
└── comparativo_v2_<dataset>_<model>.png
```

Each PNG shows two panels:
- **Left:** Training loss per epoch (mean ± std across runs)
- **Right:** Validation metric per epoch, with final test score annotated

## Reproducibility

Each independent run is seeded separately before weight initialisation, ensuring that runs within the same experiment start from different weight configurations while remaining fully reproducible across repeated executions.
