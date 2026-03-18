# Bridging Molecular Fingerprints and Graph Neural Networks through Knowledge-Guided Pooling


[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)](https://pytorch.org/)

**FPPool** is a chemistry-aware graph pooling method that uses molecular fingerprints as substructure discovery modules to guide hierarchical aggregation in graph neural networks. Instead of treating molecules as generic graphs, FPPool decomposes each molecular graph into biochemically meaningful substructure groups via bit-level fingerprint matching, then progressively aggregates them through a three-level attention mechanism (atom → substructure → fingerprint → molecule).

<p align="center">
  <img src="assets/FPPool_framework.png" width="90%" alt="FPPool Framework"/>
</p>

## Highlights

- 🧬 **Chemistry-aware pooling** — Uses molecular fingerprints (PubChem, Morgan, RDKit, etc.) to decompose molecules into meaningful functional groups before aggregation
- 📊 **Strong performance** — Outperforms global pooling (Sum, Mean, Max, Set2Set) and hierarchical pooling (TopKPool, SAGPool, DiffPool, MinCutPool) across 40 benchmarks
- 🔍 **Built-in interpretability** — Multi-scale attention weights trace predictions to specific atoms, substructures, and fingerprint types without post-hoc methods
- 🔌 **Drop-in replacement** — Compatible with any GNN backbone (GIN, GCN, GAT, GraphSAGE) as a plug-and-play pooling module
- ⚗️ **Activity cliff robustness** — Detects subtle structural modifications that drive large potency shifts in activity-cliff datasets

## Installation

```bash
git clone https://github.com/xiaocui3737/FPPooling.git
cd FPPooling
pip install -r requirements.txt
```

### Dependencies

- Python ≥ 3.8
- PyTorch ≥ 1.12
- PyTorch Geometric ≥ 2.0
- RDKit ≥ 2022.03
- NumPy, Pandas, Scikit-learn

## Quick Start

### MoleculeNet Benchmark

```bash
# Run on BBBP (classification)
python MoleculeNet_example.py --dataset bbbp --fp_type M+Rd+Pub

# Run on ESOL (regression)
python MoleculeNet_example.py --dataset esol --fp_type M+Rd+Pub
```

### MoleculeACE Benchmark (Activity Cliffs)

```bash
# Run on a specific target
python MoleculeACE_example.py --target ABL1 --fp_type M+Rd+Pub
```

### Use FPPool in Your Own Pipeline

```python
from fppcode import FPPool, compute_fingerprints

# Replace your existing pooling layer with FPPool
model = YourGNN(pooling=FPPool(
    fp_types=['Morgan', 'PubChem', 'Rdkit'],
    hidden_dim=256
))

# FPPool handles fingerprint computation and matching internally
output = model(batch)
```

## Repository Structure

```
FPPooling/
├── fppcode/                    # Core implementation
│   ├── model.py                # FPPool pooling module
│   ├── fingerprints.py         # Fingerprint computation & SMARTS matching
│   ├── gnn_backbone.py         # GNN backbone (GIN/GCN/GAT)
│   └── utils.py                # Utilities
├── MoleculeNet_example.py      # MoleculeNet training & evaluation
├── MoleculeACE_example.py      # MoleculeACE training & evaluation
├── requirements.txt            # Dependencies
├── assets/                     # Figures for README
└── README.md
```

## Method Overview

FPPool operates in three stages:

**Stage 1: GNN Message Passing** — Standard MPNN updates node features through neighborhood aggregation.

**Stage 2: Fingerprint-Guided Substructure Matching** — Multiple fingerprint types (dictionary-based, circular, path-based) decompose the molecular graph into overlapping substructure groups via bit-level SMARTS matching.

**Stage 3: Hierarchical Attention Pooling**
| Level | Operation | Granularity |
|-------|-----------|-------------|
| **InnerPool** | Attention-weighted aggregation within each substructure | Atom → Substructure |
| **InterPool** | Attention-weighted aggregation across substructures of the same FP | Substructure → Fingerprint |
| **FinalPool** | Attention-weighted aggregation across fingerprint types | Fingerprint → Molecule |

Each level produces learnable importance weights, providing built-in interpretability at every granularity.

## Supported Fingerprints

| Type | Name | Bits | Description |
|------|------|------|-------------|
| Dictionary | PubChem | 881 | Substructure key-based |
| Dictionary | MACCS | 166 | Structural key |
| Dictionary | RGroup | 2048 | R-group fragment |
| Dictionary | Fragments | 86 | Fragment statistical |
| Circular | Morgan | 1024 | Atom environment (ECFP) |
| Path | RDKit | 1024 | Bond path-based |
| Other | E-State | 79 | Electrotopological state |

## Results Summary

### MoleculeNet (10 datasets)

FPPool (M+Rd+Pub) achieves the best overall performance on **all 10 datasets** compared to 8 pooling baselines. See the paper for full results.

### MoleculeACE (30 activity-cliff datasets)

FPPool achieves the best RMSE on **23/30 datasets** and the lowest mean RMSE across all four target families (GPCR, NR, Kinase, Other).

## Interpretability

FPPool provides multi-scale explanations without post-hoc methods:

<p align="center">
  <img src="assets/interpretability.png" width="85%" alt="Multi-level interpretability"/>
</p>

- **Inner view**: Which atoms matter within each substructure?
- **Inter view**: Which substructures matter within each fingerprint?
- **Final view**: Which fingerprint types matter for this molecule?

## Citation

If you find FPPool useful in your research, please cite:

```bibtex
@article{cui2026fppool,
  title={Bridging molecular fingerprints and graph neural networks through knowledge-guided pooling},
  author={Cui, Chao and Feng, ShiHang and Song, ChenXiao and Yu, ShunTao and Lu, SongLin and Chen, YuZong and Shen, WanXiang},
  journal={Nature Computational Science},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **Wanxiang Shen** (corresponding) — [shenwx25@zju.edu.cn](mailto:shenwx25@zju.edu.cn)
- **Chao Cui** — [GitHub](https://github.com/xiaocui3737)
