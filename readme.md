### Bridging molecular fingerprints and graph neuralnetworks through knowledge-guided pooling (FPPool)

FPPool is a molecular graph pooling method that integrates **multiple molecular fingerprints** to improve **graph-level representation learning** and provide **interpretable substructure-level explanations**.

## Features

- Multi-fingerprint guided pooling for molecular graphs  
- Compatible with common GNN backbones (e.g., GIN/GCN/GraphSAGE)  
- Ready-to-run examples on **MoleculeNet** and **MoleculeACE**  
- Interpretability support: highlights key fragments/substructures contributing to predictions  

## Repository Structure

- `fppcode/` : core implementation of FPPool  
- `MoleculeNet_example.py` : training/evaluation example on MoleculeNet tasks  
- `MoleculeACE_example.py` : training/evaluation example on MoleculeACE benchmark  
- `readme.md` : this document  

## Quick Start

### 1) Install

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>
pip install -r requirements.txt
```

### 2) Run examples

**MoleculeNet**

```bash
python MoleculeNet_example.py
```

**MoleculeACE**

```bash
python MoleculeACE_example.py
```

## Notes

- Default hyperparameters and dataset configs are defined inside the example scripts.
- To use FPPool in your own pipeline, import from `fppcode/` and replace the original pooling module.