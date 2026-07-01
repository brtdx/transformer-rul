#!/usr/bin/env python3
"""Training + early stopping for RULTransformer (single fold)."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.transformer import RULTransformer, weighted_huber_loss

MODEL_REGISTRY = {
    'std': RULTransformer,
}


class EarlyStopping:
    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def split_train_val(train_loader, val_fraction=0.2, seed=42):
    """Per-cell chronological block split for early stopping (Faz B1).

    See root trainer.split_train_val for rationale (temporal leakage fix).
    """
    ds = train_loader.dataset
    X = ds.X
    y = ds.y
    boundaries = getattr(ds, 'cell_boundaries', None)
    if boundaries is None:
        boundaries = [(0, len(ds))]
    train_idx, val_idx = [], []
    for start, end in boundaries:
        block_len = end - start
        n_val = max(int(block_len * val_fraction), 1)
        val_idx.extend(range(end - n_val, end))
        train_idx.extend(range(start, end - n_val))
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    return (torch.utils.data.TensorDataset(X[train_idx], y[train_idx]),
            torch.utils.data.TensorDataset(X[val_idx], y[val_idx]))


def train_one_fold(train_loader, device='cpu',
                   max_epochs=50, lr=5e-3, weight_decay=1e-5,
                   patience=10, batch_size=64, verbose=True,
                   model_kwargs=None, seed=42, model_name='std'):
    # Build train/val split for early stopping
    train_sub, val_sub = split_train_val(train_loader, val_fraction=0.2, seed=seed)
    train_dl = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

    if model_kwargs is None:
        model_kwargs = {}
    # Auto-detect n_features and lookback from data (handles feature-set/seq-len changes)
    sample_x, _ = next(iter(train_dl))
    n_features = sample_x.shape[-1]
    lookback = sample_x.shape[1]
    model_kwargs = {**model_kwargs, 'n_features': n_features}
    ffn_dim = model_kwargs.get('ffn_dim', 2 * model_kwargs.get('d_model', 32))
    model_cls = MODEL_REGISTRY[model_name]  # 'std' or 'itransformer' must be in registry
    extra = {}
    if model_name == 'itransformer':
        extra['lookback'] = lookback
    model = model_cls(
        n_features=model_kwargs.get('n_features', 15),
        d_model=model_kwargs.get('d_model', 32),
        n_heads=model_kwargs.get('n_heads', 2),
        n_layers=model_kwargs.get('n_layers', 2),
        ffn_dim=ffn_dim,
        dropout=model_kwargs.get('dropout', 0.1),
        horizons=model_kwargs.get('horizons', (5, 10, 20)),
        **extra,
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max_epochs, eta_min=1e-5)

    es = EarlyStopping(patience=patience, min_delta=1e-3)
    history = {'train': [], 'val': []}

    for epoch in range(max_epochs):
        model.train()
        running = 0.0; n = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            pred = model(xb)
            loss = weighted_huber_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += float(loss.item()) * len(yb)
            n += len(yb)
        sched.step()
        train_loss = running / n

        model.eval()
        v_run = 0.0; v_n = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                l = weighted_huber_loss(pred, yb)
                v_run += float(l.item()) * len(yb)
                v_n += len(yb)
        val_loss = v_run / v_n
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        es(val_loss, model)
        if verbose and (epoch % 2 == 0 or es.early_stop or epoch == max_epochs - 1):
            print(f"  epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  LR={optim.param_groups[0]['lr']:.2e}  ES={es.counter}/{patience}")
        if es.early_stop:
            if verbose:
                print(f"  early stop @ epoch {epoch}")
            break
    es.restore(model)
    return model, history