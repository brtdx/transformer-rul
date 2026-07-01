# Transformer-RUL

Cycle-level RUL (Remaining Useful Life) prediction for lithium-ion batteries using the **standard Transformer encoder** (Vaswani et al., NeurIPS 2017).

## Overview

This repository contains a PyTorch implementation of the standard Transformer **encoder** architecture adapted for cycle-level SoH regression and RUL estimation on a 4-cell battery aging dataset (Ch1/Ch2/Ch3/Ch6).

**Key idea:** each cycle in the lookback window is a token. Multi-head self-attention captures temporal dependencies across cycles. This is the encoder-only Transformer applied to time-series regression.

Reference: Vaswani et al., "Attention Is All You Need", NeurIPS 2017. arXiv:1706.03762.
URL: https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html

### Architectural notes (deviations from the 2017 paper)

This implementation uses a **modernized** Transformer encoder:

| Component | Vaswani 2017 | This impl | Notes |
|---|---|---|---|
| Positional encoding | Sinusoidal | Sinusoidal | Exact match |
| Multi-Head Attention | ✓ | ✓ | PyTorch built-in |
| Feed-Forward dim | 4×d_model | 2×d_model | Smaller for our small d=8 |
| Activation | ReLU | GELU | BERT-style modern variant |
| LayerNorm order | Post-LN | Pre-LN | Stable training variant (2020) |
| Decoder | ✓ (seq2seq) | ✗ | Encoder-only for regression |

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

---

## Changelog (Done / TODO)

> This section tracks engineering improvements made (or planned) after an external
> code review. It is split into two phases: **Phase A** (done, no re-train) and
> **Phase B** (in progress, requires re-train). Latest status is tracked here.

### Phase A — Done (no re-train required)

- **A1 — Normalization NaN safety**: `nan_to_num` appended to
  `compute_fold_stats`. Dead (all-NaN) feature columns now yield `mean=0,
  std=0` instead of NaN. `norm_stats.npz` no longer contains NaNs. Model
  behaviour verified identical (md5 check).
- **A2 — Exponential RUL extrapolation**: a log-linear (exponential) fit was
  added to `derive_rul_from_soh`. For cells that never reach EOL, both a
  linear and an exponential crossing are computed, and the pessimistic (safe)
  one is selected via `min(rul_lin, rul_exp)`. Battery degradation accelerates
  (knee) near EOL → linear fit is over-optimistic, so the minimum keeps the
  estimate on the safe (under-estimate) side.
- **A3 — Inference clamp**: `predict_test_pack` now applies `np.clip(out, 0, 1)`
  instead of a sigmoid (inference-only). Training gradients are unaffected →
  HuberLoss (delta=0.01) stays robust in the linear region; the clamp is purely
  a physical-range safety guard.
- **A4 — Re-evaluation script**: `reeval.py` re-evaluates existing `model.pt`
  checkpoints with the clean `norm_stats` + A2/A3 fixes without any re-train.
  For iTransformer, predictions.npz md5 and PNGs came out byte-identical.

### Phase B — In progress (requires re-train)

- **B0 — Synchronization (DONE)**: `audit_leakage.py` path fixes (after
  Bug-C result-directory migration), all sweep scripts synced with Phase A
  (clamp + exp fit), and a latent `count_parameters` import bug fixed (it was
  missing from the root `trainer.py`). Audit now passes 20/20.
- **B1 — Per-cell chronological val split (DONE)**: `split_train_val` now holds
  out the **last 20%** of each cell's windows as validation (instead of
  shuffling). With stride=1 sliding windows, consecutive windows share 19/20
  cycles (~95% overlap). A random shuffle lets these near-duplicates land in
  both train and val → artificially low val loss → biased early stopping.
  Audit CHECK 5: stride-neighbour ratio dropped **97.9% → 1.6%**.
- **B2 — Backup (DONE)**: `results/itransformer/` → `baseline_v1/`, sweep
  CSVs → `sweep/baseline_v1/`.
- **B3 — Full re-train (TODO)**: (1) std sweep (12 configs × 4 folds),
  (2) std round 2, (3) iTransformer sweep (6 configs), (4) std best-config
  4-fold LOCO, (5) iTransformer best-config 4-fold LOCO, (6) cell_3 multi-seed
  (5 seeds) + ensemble. Estimated total runtime: ~20-25 min.
- **B4 — Validation (TODO)**: confirm audit 19/19 PASS, update both
  `architecture.md` files with a "Phase B per-cell chronological val split"
  section, and build a v1-vs-v2 metrics comparison table. Re-train metrics may
  shift — this is the honest report (leak-free early stopping yields more
  trustworthy numbers).

### Notes (what may not change)

- **Feature set (15) is FIXED** — no Phase A/B touch touches the feature set,
  to preserve a fair std-vs-iTransformer comparison.
- **iTransformer IT_d32_L1 (d=32, L=1, H=2)** will likely remain the winning
  config. If the winning config changes, the explanation is: "the temporal
  leakage fix brought more robust early stopping, allowing less overfitting."
