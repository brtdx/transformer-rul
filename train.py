#!/usr/bin/env python3
"""4-fold LOCO training + evaluation for RUL prediction.

Supports two transformer backends via MODEL_NAME:
  - 'std'         : RULTransformer (standard time-token attention)
  - 'itransformer': iTransformer (Liu et al., ICLR 2024, inverted varate-token attention)

For each LOCO fold (test_cell in {cell_1, cell_2, cell_3, cell_6}):
  1. Build fold loaders (train=other 3 cells, test=held-out)
  2. Train transformer (AdamW + Weighted HuberLoss + early stopping)
  3. Save model state_dict and fold metadata to results/fold_<test_cell>/
  4. Evaluate multi-horizon SoH MAE/RMSE on the held-out cell
  5. Derive RUL (residual cycles to SoH<fallback 80%) from predicted SoH trajectory

Outputs:
  results/fold_<cell>/model.pt
  results/fold_<cell>/history.npz
  results/fold_<cell>/predictions.npz  (anchor_cycle, SoH_true[k+5/10/20], avg,
                                         SoH_pred[k+5/10/20], RUL_true, RUL_pred)
  results/summary.json   (mean+std metrics across 4 folds)
  results/soh_trajectories[_itr].png  (predicted vs true SoH per cell)
  results/rul_predictions[_itr].png   (RUL_true vs RUL_pred)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import (CELL_IDS, HORIZONS, WINDOW, make_fold_loaders,
                          make_test_prediction_pack)
from model.trainer import train_one_fold

RESULTS = Path('/home/bbb/Desktop/rul/results') / 'std'
RESULTS.mkdir(parents=True, exist_ok=True)
DEVICE = 'cpu'
EOL_SOH = 80.0   # original SoH scale (0-100%)
EOL_SOH_NORM = EOL_SOH / 100.0  # = 0.80 in normalized [0,1] scale
SOH_SCALE = 100.0

# Best config from iTransformer sweep (IT_d32_L1):
#   d=32, L=1, H=2, do=0.1, 11,395 params.
# iTransformer (Liu et al., ICLR 2024) outperforms standard RULTransformer on
# cross-profile cell_3 (DST): h20 MAE 3.5% vs 14.1% (4x improvement), while
# keeping standard cells (cell_1/2/6) at h20~6.2% (comparable to R2_d8's 6.4%).
MODEL_NAME = 'std'  # Standard RULTransformer (Vaswani et al., 2017)
MODEL_KWARGS = {'d_model': 32, 'n_layers': 1, 'n_heads': 2, 'dropout': 0.1,
                'weight_decay': 1e-5, 'lr': 5e-3, 'patience': 10}
PLOT_SUFFIX = '_itr' if MODEL_NAME == 'itransformer' else ''


def predict_test_pack(model, pack, device='cpu'):
    """Run model on full-trace windows. Returns predicted SoH (normalized [0,1]) per horizon."""
    model.eval()
    X = torch.from_numpy(pack['X']).to(device)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            pred = model(X[i:i + 128]).cpu().numpy()  # (b, 3) normalized
            preds.append(pred)
    out = np.concatenate(preds, axis=0)  # (N_anchors, 3)
    # SoH physical bounds [0,1]: inference-only clamp instead of sigmoid.
    # Training gradients unaffected (model heads untouched) -> no re-train needed.
    return np.clip(out, 0.0, 1.0)


def derive_rul_from_soh(soh_pred_norm, threshold_norm=EOL_SOH_NORM):
    """Given predicted SoH series (normalized), for each anchor compute how many steps
    until SoH falls below threshold.

    If the predicted trajectory crosses threshold within available data → direct step count.
    If it never crosses -> extrapolation (linear + exponential fit, take the pessimistic one).
    Battery degradation accelerates near EOL (knee point): exponential curve captures this,
    linear fit stays optimistic; min(rul_lin, rul_exp) takes the safe side (under-estimate RUL).
    """
    n = len(soh_pred_norm)
    rul = np.full(n, np.nan, dtype=np.float32)
    extrap_win = min(30, max(5, n // 4))

    for i in range(n):
        future = soh_pred_norm[i:]
        below = np.where(future < threshold_norm)[0]
        if len(below) > 0:
            rul[i] = float(below[0])
        else:
            if len(future) >= extrap_win:
                recent = future[-extrap_win:]
                x = np.arange(len(recent), dtype=np.float32)
                # Linear fit: SoH = a1*x + b1
                slope_lin, intercept_lin = np.polyfit(x, recent, 1)
                rul_lin = np.nan
                if slope_lin < -1e-8:
                    x_cross = (threshold_norm - intercept_lin) / slope_lin
                    offset = len(future) - extrap_win + x_cross
                    if offset > 0:
                        rul_lin = float(offset)
                # Exponential fit (log-linear): log(SoH)=a2*x+b2 -> SoH=exp(b2)*exp(a2*x)
                rul_exp = np.nan
                if np.all(recent > 1e-4):
                    log_recent = np.log(recent)
                    slope_exp, intercept_exp = np.polyfit(x, log_recent, 1)
                    if slope_exp < -1e-8:
                        x_cross = (np.log(threshold_norm) - intercept_exp) / slope_exp
                        offset = len(future) - extrap_win + x_cross
                        if offset > 0:
                            rul_exp = float(offset)
                # Pessimistic (safe) RUL: min(rul_lin, rul_exp)
                candidates = [v for v in (rul_lin, rul_exp) if not np.isnan(v)]
                if candidates:
                    rul[i] = float(min(candidates))
    return rul


def evaluate_fold(model, pack):
    """Return dict of metrics (in % SoH scale) and arrays (in original 0-100% for plotting)."""
    pred_norm = predict_test_pack(model, pack, device=DEVICE)  # (N, 3) normalized
    y_norm = pack['y']         # normalized [0,1]
    mask = pack['mask']
    cycle_ids = pack['cycle_ids']
    horizon_names = [f'h{h}' for h in HORIZONS]

    # Convert to original 0-100% scale for metrics and plotting
    pred_pct = pred_norm * SOH_SCALE    # (N, 3)
    y_pct = y_norm * SOH_SCALE

    metrics = {}
    for k, h in enumerate(HORIZONS):
        valid = mask[:, k]
        p = pred_pct[valid, k]
        t = y_pct[valid, k]
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        metrics[f'SoH_MAE_{horizon_names[k]}'] = mae
        metrics[f'RMSE_{horizon_names[k]}'] = rmse

    # Trajectory: use h=20 prediction (captures degradation trend better than h=5)
    soh_trajectory_pred = pred_pct[:, 2]   # h=20, original % scale
    soh_trajectory_true = pack['SoH_true_orig'][cycle_ids - 1]  # original 0-100%

    # RUL derivation (in normalized space for threshold comparison, using h=20 trajectory)
    soh_pred_norm_traj = pred_norm[:, 2]   # h=20 predicted SoH (normalized)
    rul_pred = derive_rul_from_soh(soh_pred_norm_traj, threshold_norm=EOL_SOH_NORM)

    # Ground-truth RUL from original SoH trajectory
    soh_true_full = pack['SoH_true_orig']  # original 0-100%
    rul_true = np.full(len(cycle_ids), np.nan, dtype=np.float32)
    for i, c in enumerate(cycle_ids):
        below = np.where(soh_true_full[c - 1:] < EOL_SOH)[0]
        if len(below) > 0:
            rul_true[i] = float(below[0])

    # RUL metrics where both available
    valid = ~np.isnan(rul_pred) & ~np.isnan(rul_true)
    if valid.sum() > 0:
        metrics['RUL_MAE'] = float(np.mean(np.abs(rul_pred[valid] - rul_true[valid])))
        metrics['RUL_RMSE'] = float(np.sqrt(np.mean((rul_pred[valid] - rul_true[valid]) ** 2)))
        metrics['RUL_n_valid'] = int(valid.sum())
    else:
        metrics['RUL_MAE'] = float('nan')
        metrics['RUL_RMSE'] = float('nan')
        metrics['RUL_n_valid'] = 0

    return {
        'metrics': metrics,
        'pred_pct': pred_pct,          # (N, 3) original %
        'y_pct': y_pct,                # (N, 3) original %
        'mask': mask,
        'cycle_ids': cycle_ids,
        'SoH_pred_trajectory': soh_trajectory_pred,   # % scale
        'SoH_true_trajectory': soh_trajectory_true,    # % scale
        'rul_pred': rul_pred,
        'rul_true': rul_true,
    }


def main():
    print("=== 4-fold LOCO RUL Transformer training ===")
    fold_results = {}
    for test_cell in CELL_IDS:
        print(f"\n>>> Fold: test={test_cell}")
        train_loader, test_pack, mean, std, feat_names = make_fold_loaders(test_cell, batch=64)
        model, history = train_one_fold(train_loader, device=DEVICE, verbose=True,
                                        max_epochs=50,
                                        lr=MODEL_KWARGS['lr'],
                                        weight_decay=MODEL_KWARGS['weight_decay'],
                                        patience=MODEL_KWARGS['patience'],
                                        model_kwargs={k: v for k, v in MODEL_KWARGS.items()
                                                      if k in ['d_model', 'n_layers', 'n_heads', 'dropout']},
                                        model_name=MODEL_NAME)
        result = evaluate_fold(model, test_pack)

        # Save artifacts
        fold_dir = RESULTS / f'fold_{test_cell}'
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), fold_dir / 'model.pt')
        np.savez(fold_dir / 'history.npz', train=history['train'], val=history['val'])
        np.savez(fold_dir / 'predictions.npz',
                 cycle_ids=result['cycle_ids'], pred_pct=result['pred_pct'],
                 y_pct=result['y_pct'], mask=result['mask'],
                 SoH_pred_trajectory=result['SoH_pred_trajectory'],
                 SoH_true_trajectory=result['SoH_true_trajectory'],
                 rul_pred=result['rul_pred'], rul_true=result['rul_true'])
        # Save normalization stats so the model can be re-used on Ch2_3 later
        np.savez(fold_dir / 'norm_stats.npz', mean=mean, std=std, features=feat_names)
        fold_results[test_cell] = result['metrics']
        print(f"  metrics: {result['metrics']}")

    # Build summary.json
    summary = {}
    # Per-horizon SoH metrics
    for h_name in [f'h{h}' for h in HORIZONS]:
        maes = [fold_results[c][f'SoH_MAE_{h_name}'] for c in CELL_IDS]
        rmses = [fold_results[c][f'RMSE_{h_name}'] for c in CELL_IDS]
        summary[f'SoH_MAE_{h_name}_mean'] = float(np.mean(maes))
        summary[f'SoH_MAE_{h_name}_std'] = float(np.std(maes))
        summary[f'RMSE_{h_name}_mean'] = float(np.mean(rmses))
        summary[f'RMSE_{h_name}_std'] = float(np.std(rmses))
    rul_maes = [fold_results[c].get('RUL_MAE', np.nan) for c in CELL_IDS]
    summary['RUL_MAE_mean'] = float(np.nanmean(rul_maes))
    summary['RUL_MAE_std'] = float(np.nanstd(rul_maes))
    summary['model_name'] = MODEL_NAME
    summary['model_kwargs'] = MODEL_KWARGS
    summary['per_fold'] = fold_results
    with open(RESULTS / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n=== Summary (4-fold LOCO) ===")
    print(json.dumps(summary, indent=2, default=str))

    # ---- Plots ----
    # 1. Predicted vs true SoH trajectory per cell (4 subplots)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    for ax, c in zip(axes.flat, CELL_IDS):
        npz = np.load(RESULTS / f'fold_{c}' / 'predictions.npz')
        cyc = npz['cycle_ids']
        st = npz['SoH_true_trajectory']
        sp = npz['SoH_pred_trajectory']
        ax.plot(cyc, st, 'k-', label='True', linewidth=1)
        ax.plot(cyc, sp, 'r--', label='Pred (h=20)', linewidth=1)
        ax.axhline(EOL_SOH, color='gray', linestyle=':', label='EOL 80%')
        ax.set_title(f'{c} (test fold)')
        ax.set_xlabel('Cycle')
        ax.set_ylabel('SoH (%)')
        ax.legend(loc='lower left', fontsize=8)
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / f'soh_trajectories{PLOT_SUFFIX}.png', dpi=120)
    plt.close()

    # 2. RUL scatter (true vs pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    all_true = []; all_pred = []
    for c in CELL_IDS:
        npz = np.load(RESULTS / f'fold_{c}' / 'predictions.npz')
        valid = ~np.isnan(npz['rul_pred']) & ~np.isnan(npz['rul_true'])
        if valid.sum() > 0:
            all_true.append(npz['rul_true'][valid])
            all_pred.append(npz['rul_pred'][valid])
    if all_true:
        all_true_arr = np.concatenate(all_true)
        all_pred_arr = np.concatenate(all_pred)
        ax.scatter(all_true_arr, all_pred_arr, alpha=0.6, s=12)
        lims = [0, max(all_true_arr.max(), all_pred_arr.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel('True RUL (cycles)')
        ax.set_ylabel('Predicted RUL (cycles)')
        ax.set_title('RUL: true vs predicted (4-fold LOCO, valid anchors only)')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS / f'rul_predictions{PLOT_SUFFIX}.png', dpi=120)
    plt.close()

    print(f"\nDone. Results saved under {RESULTS}/")
    print(f"  - summary.json")
    print(f"  - soh_trajectories{PLOT_SUFFIX}.png")
    print(f"  - rul_predictions{PLOT_SUFFIX}.png")


if __name__ == '__main__':
    main()