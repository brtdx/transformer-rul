# Transformer-RUL

Cycle-level RUL (Remaining Useful Life) prediction for lithium-ion batteries using the **standard Transformer** (Vaswani et al., NeurIPS 2017).

## Overview

This repository contains a PyTorch implementation of the original Transformer architecture adapted for cycle-level SoH regression and RUL estimation on a 4-cell battery aging dataset (Ch1/Ch2/Ch3/Ch6, 20°C climate chamber).

**Key idea:** each cycle in the lookback window is a token. Multi-head self-attention captures temporal dependencies across cycles. This is the classical Transformer applied to time-series regression.

Reference: Vaswani et al., "Attention Is All You Need", NeurIPS 2017.

## Project structure

```
.
├── train.py                  # 4-fold LOCO training + evaluation entrypoint
├── model/
│   ├── transformer.py        # RULTransformer model (96 lines)
│   └── trainer.py            # Training loop with weighted HuberLoss
├── data/
│   └── dataset.py            # Sliding-window dataset + z-score normalization + LOCO fold loader
└── features/
    └── extract_features.py   # 15 physics-based features extracted from DuckDB/HDF5
```

## Configuration

- embed_dim (d): 8
- layers (L): 2
- heads (H): 2
- lookback (T): 20 cycles
- features: 15 (no Cycle_number)

## How to run

```bash
# 1. Extract features from DuckDB/HDF5
python3 features/extract_features.py

# 2. Train + evaluate (4-fold LOCO)
python3 train.py
```

Outputs:
- `results/fold_<cell>/model.pt`
- `results/fold_<cell>/predictions.npz`
- `results/summary.json`
- `results/soh_trajectories.png`
- `results/rul_predictions.png`
