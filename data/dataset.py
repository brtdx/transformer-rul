#!/usr/bin/env python3
"""Dataset construction for RUL Transformer (4-fold LOCO).

For each fold k (k=1..4):
  train_cells = all cells except test_cell[k]
  test_cell    = test_cell[k]

Per cell:
  1. z-score features using statistics computed from TRAIN cells only (no leakage:
     normalization stats come from the 3 training cells, applied to the held-out cell)
  2. Form sliding windows (window L=20, stride=1):
       X_win[i]  = z_X[i : i+L]            shape (L, F)
       y_win[i]  = [SoH[i+L-1+h] for h in horizons]  shape (3,)
     Truncated from the end to keep valid future horizons.
  3. Concatenate windows from all train cells into the fold's train set.

Output: helper functions used by train.py
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

FEATURES_PATH = str(Path(__file__).parent.parent.parent / 'features_all_cells.hdf5')
CELL_IDS = ['cell_1', 'cell_2', 'cell_3', 'cell_6']
HORIZONS = [5, 10, 20]
WINDOW = 20
SEED = 42
SOH_SCALE = 100.0  # SoH stored as 0-100%; normalized target = SoH / SOH_SCALE -> [0,1]


def load_raw(features_path=FEATURES_PATH):
    """Return dict {cell_id: (X, SoH_norm, CycleID, SoH_orig)} and feature names.

    SoH_norm: SoH / 100 -> [0,1] for model training
    SoH_orig: original 0-100% for metrics and plotting
    """
    data = {}
    with h5py.File(features_path, 'r') as f:
        feature_names = [s.decode('utf-8') for s in f.attrs['feature_names']]
        horizons = list(f.attrs['horizons'])
        for cid in CELL_IDS:
            soh_orig = f[cid]['SoH'][:].astype(np.float32)
            data[cid] = (f[cid]['X'][:].astype(np.float32),
                         (soh_orig / SOH_SCALE).astype(np.float32),  # normalized [0,1]
                         f[cid]['CycleID'][:].astype(np.int32),
                         soh_orig)  # original 0-100%
    return data, feature_names, horizons


def compute_fold_stats(train_cells_data):
    """Per-feature z-score mean/std over ALL cycles of the train cells (pooled).

    NaNs are replaced with the global mean of that feature (computed ignoring NaNs)
    BEFORE computing pooled stats. This guarantees no NaN survives into the model.
    """
    Xs = []
    for cid in train_cells_data:
        X = train_cells_data[cid][0].copy()
        # Replace NaN with per-feature nanmean
        col_nanmean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_nanmean, inds[1])
        Xs.append(X)
    pooled = np.concatenate(Xs, axis=0)
    mean = pooled.mean(axis=0)
    std = pooled.std(axis=0)
    std[std < 1e-6] = 1.0  # sabit-ama-real kolonlar için div-by-zero koruması (std=0 → 1.0)
    # Dead (all-NaN-train) kolon güvenliği: norm_stats.npz NaN içermesin.
    # NaN mean/std → 0: zscore (X-0)/0=inf → nan_to_num → 0 (mevcut davranış birebir korunur,
    # hem train hem test dead kolonu 0'a düşer; model "dead feature" öğrenir, OOD engellenir).
    # Bazen: dead kolonlar için std guard NaN<1e-6=False atlar, std NaN kalır → nan_to_num std=0.
    # Gerçek sabit kolonlar için std=0 → guard 1.0'a yükseltir (NaN olmayan, gerçek 0).
    mean = np.nan_to_num(mean, nan=0.0)
    std  = np.nan_to_num(std,  nan=0.0)
    return mean.astype(np.float32), std.astype(np.float32)


def zscore(X, mean, std):
    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    return Xs.astype(np.float32)


def build_windows(X_z, SoH, L=WINDOW, horizons=HORIZONS):
    """Construct sliding-window inputs and multi-horizon SoH labels.

    For each anchor i such that i+L-1+max(horizons) < len(SoH):
      X_win = X_z[i : i+L]                    (L, F)
      y_win = [SoH[i+L-1+h] for h in horizons] (len(horizons),)
    """
    n = len(SoH)
    max_h = max(horizons)
    last = n - L - max_h  # last valid anchor
    if last < 0:
        return np.zeros((0, L, X_z.shape[1]), dtype=np.float32), np.zeros((0, len(horizons)), dtype=np.float32)
    X_windows = np.stack([X_z[i:i + L] for i in range(last + 1)], axis=0)
    y_windows = np.stack([SoH[i + L - 1 + np.array(horizons)] for i in range(last + 1)], axis=0)
    return X_windows.astype(np.float32), y_windows.astype(np.float32)


def build_full_trace(X_z, SoH, L=WINDOW, horizons=HORIZONS):
    """For evaluation: build windows along the ENTIRE cycle trajectory (no end-truncation).

    Anchors where the future SoH horizon is unavailable get a NaN label (ignored in metrics).
    Returns X_windows (N, L, F), y_windows (N, H), valid_mask (N, H).
    """
    n = len(SoH)
    max_h = max(horizons)
    X_windows = []
    y_windows = []
    valid_mask = []
    for i in range(n - L + 1):
        X_windows.append(X_z[i:i + L])
        y = np.full(len(horizons), np.nan, dtype=np.float32)
        mask = np.zeros(len(horizons), dtype=bool)
        for k, h in enumerate(horizons):
            idx = i + L - 1 + h
            if idx < n:
                y[k] = SoH[idx]
                mask[k] = True
        y_windows.append(y)
        valid_mask.append(mask)
    return (np.array(X_windows, dtype=np.float32),
            np.array(y_windows, dtype=np.float32),
            np.array(valid_mask, dtype=bool))


class CycleWindowDataset(Dataset):
    def __init__(self, X_windows, y_windows, cell_boundaries=None):
        self.X = torch.from_numpy(X_windows)
        self.y = torch.from_numpy(y_windows)
        # Per-cell contiguous index ranges [(start, end), ...] in chronological
        # order. Set by make_fold_loaders so split_train_val can carve a
        # per-cell chronological val block and avoid temporal leakage between
        # stride-1 sliding windows (Faz B1).
        self.cell_boundaries = cell_boundaries

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_fold_loaders(test_cell_id, batch=64, features_path=FEATURES_PATH):
    """Return train_loader, test_raw (unbatched dict with windows + cycle ids)."""
    data, feature_names, horizons = load_raw(features_path)
    train_ids = [c for c in CELL_IDS if c != test_cell_id]
    mean, std = compute_fold_stats({c: data[c] for c in train_ids})

    # ---- Train: concat windows from all train cells (chronological, no shuffle) ----
    # Per-cell block boundaries recorded for leak-safe val split (Faz B1).
    X_tr_all, y_tr_all = [], []
    boundaries = []
    offset = 0
    for cid in train_ids:
        Xz = zscore(data[cid][0], mean, std)
        Xw, yw = build_windows(Xz, data[cid][1])
        X_tr_all.append(Xw)
        y_tr_all.append(yw)
        boundaries.append((offset, offset + len(Xw)))
        offset += len(Xw)
    X_train = np.concatenate(X_tr_all, axis=0)
    y_train = np.concatenate(y_tr_all, axis=0)

    train_ds = CycleWindowDataset(X_train, y_train, cell_boundaries=boundaries)
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=False)

    # ---- Test: keep per-cycle (un-batched), full trajectory, anchored windows ----
    Xz_test = zscore(data[test_cell_id][0], mean, std)
    Xw_test, yw_test, mask_test = build_full_trace(Xz_test, data[test_cell_id][1])
    test_pack = {
        'cell_id': test_cell_id,
        'X': Xw_test,
        'y': yw_test,          # normalized [0,1]
        'mask': mask_test,
        'cycle_ids': data[test_cell_id][2][WINDOW - 1:],
        'SoH_true': data[test_cell_id][1],       # normalized [0,1] — for loss-aware metrics
        'SoH_true_orig': data[test_cell_id][3],   # original 0-100% — for plots/RUL
    }
    return train_loader, test_pack, mean, std, feature_names


def make_test_prediction_pack(cell_id, mean, std, features_path=FEATURES_PATH):
    """Rebuild a test pack (full-trace windows) for a given cell using existing stats."""
    data, _, _ = load_raw(features_path)
    Xz = zscore(data[cell_id][0], mean, std)
    Xw, yw, mask = build_full_trace(Xz, data[cell_id][1])
    return {
        'cell_id': cell_id,
        'X': Xw,
        'y': yw,
        'mask': mask,
        'cycle_ids': data[cell_id][2][WINDOW - 1:],
        'SoH_true': data[cell_id][1],            # normalized
        'SoH_true_orig': data[cell_id][3],        # original 0-100%
    }


if __name__ == '__main__':
    # Smoke test: build all 4 folds, print shapes, sanity-check stats
    for test_cell in CELL_IDS:
        tl, tp, m, s, fns = make_fold_loaders(test_cell, batch=64)
        Xb, yb = next(iter(tl))
        print(f"  Test={test_cell}: train windows={len(tl.dataset)}, "
              f"batch X={Xb.shape} y={yb.shape}, "
              f"test windows={tp['X'].shape}, test cycle_ids {tp['cycle_ids'][0]}..{tp['cycle_ids'][-1]}")
    print("Dataset module OK.")