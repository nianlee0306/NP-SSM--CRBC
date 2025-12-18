# NP-SSM--CRBC

This repository contains the implementation of a **Neural Passive State-Space Complete Radiation Boundary Condition (NP-SSM-CRBC)** for time-domain electromagnetic simulations.

The code supports data generation using Meep and model training using PyTorch.

---

## System Requirements

⚠️ **Platform limitation**

This project is supported **only on**:

- **Linux**
- **Windows Subsystem for Linux 2 (WSL2)**

---

## Environment Setup

### 1. Conda Environment (Required)

All dependencies should be installed via **Conda / Miniconda**.

Key requirements:
- Python ≥ 3.9
- Conda-forge channel enabled
- `pymeep` (Meep Python bindings)
- `mpi4py`
- `numpy`, `tqdm`
- `pytorch` (CPU or CUDA, depending on use case)

Example:
```bash
conda create -n pml_meep python=3.10 -y
conda activate pml_meep
conda install -c conda-forge pymeep mpi4py numpy tqdm -y
