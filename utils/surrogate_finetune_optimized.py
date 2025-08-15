
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from utils.networks import FCNetwork_mnist, ClientFuncRegressionModel_rep
from utils.utils_neigh import build_kNN_graph


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda")
    return torch.device("cpu")


def _as_tensor(x, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def set_torch_deterministic(seed: Optional[int] = None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def positional_encoding(x: torch.Tensor, L: int = 4) -> torch.Tensor:
    # NeRF-style positional encoding, vectorized and device/dtype-safe.
    if L <= 0:
        return x
    freqs = (2.0 ** torch.arange(L, device=x.device, dtype=x.dtype)).view(1, L, 1)
    angles = x.unsqueeze(1) * freqs
    return torch.cat([x, torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)], dim=1)


def build_surrogate_dataset_neg(
    Z_query_nonNN: np.ndarray,
    Z_local: np.ndarray,
    noise_in_estimator: float = 1.0,
    eps: float = 1.0,
    clamp_low: float = 1e-4,
    clamp_high: float = 1.0,
    drop_largest: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    # Vectorized negative-loss builder.
    Zq = np.asarray(Z_query_nonNN, dtype=np.float32)
    Zl = np.asarray(Z_local, dtype=np.float32)

    a2 = np.sum(Zq * Zq, axis=1, keepdims=True)
    b2 = np.sum(Zl * Zl, axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (Zq @ Zl.T)
    np.maximum(d2, 0.0, out=d2)

    if drop_largest > 0 and Zl.shape[0] > drop_largest:
        kth = Zl.shape[0] - drop_largest
        idx = np.argpartition(d2, kth=kth-1, axis=1)[:, :kth]
        rows = np.arange(d2.shape[0])[:, None]
        d2_kept = d2[rows, idx]
    else:
        d2_kept = d2

    estimator = 1.0 / (1.0 + noise_in_estimator * (d2_kept + eps))
    loss_terms = -np.log(np.clip(1.0 - estimator, clamp_low, clamp_high))
    y = np.mean(loss_terms, axis=1, dtype=np.float32)[:, None]

    return Zq.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


class ArrayDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.shape[0] == y.shape[0]
        self.X, self.y = X, y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]


@dataclass
class TrainConfig:
    lr_new: float = 1e-3
    lr_finetune: float = 1e-4
    batch_size: int = 128
    max_epochs_new: int = 40
    max_epochs_finetune: int = 8
    early_patience: int = 4
    val_frac: float = 0.1
    grad_clip_norm: float = 1.0
    num_workers: int = 0
    pin_memory: bool = False
    pe_bands: int = 4
    validate_every: int = 2
    seed: Optional[int] = None


def train_surrogate_rep(
    train_set_x: np.ndarray,
    train_set_y: np.ndarray,
    regressor_buffer=None,
    device: Optional[torch.device] = None,
    cfg: TrainConfig = TrainConfig(),
) -> nn.Module:
    device = device or get_device()
    set_torch_deterministic(cfg.seed)

    X = torch.from_numpy(train_set_x)
    y = torch.from_numpy(train_set_y)
    X = positional_encoding(X, L=cfg.pe_bands)
    e_dim = int(X.shape[1])

    n = len(y)
    n_val = max(1, int(round(n * cfg.val_frac)))
    rng_seed = cfg.seed if cfg.seed is not None else int(time.time())
    rng = np.random.default_rng(rng_seed)
    val_idx = rng.choice(n, size=n_val, replace=False)
    train_mask = np.ones(n, dtype=bool)
    train_mask[val_idx] = False

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_idx], y[val_idx]

    loader = DataLoader(
        ArrayDataset(X_tr, y_tr),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    regressor = ClientFuncRegressionModel_rep(e_dim=e_dim, output_dim=1)
    if regressor_buffer is not None:
        regressor_buffer.seek(0)
        state_dict = torch.load(regressor_buffer, map_location="cpu")
        regressor.load_state_dict(state_dict)
        lr, max_epochs = cfg.lr_finetune, cfg.max_epochs_finetune
    else:
        lr, max_epochs = cfg.lr_new, cfg.max_epochs_new

    regressor.to(device)
    opt = torch.optim.Adam(regressor.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in regressor.state_dict().items()}
    bad = 0

    for epoch in range(max_epochs):
        regressor.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=False)
            yb = yb.to(device, non_blocking=False)
            opt.zero_grad(set_to_none=True)
            pred = regressor(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(regressor.parameters(), cfg.grad_clip_norm)
            opt.step()

        if (epoch % max(1, cfg.validate_every)) != 0 and (epoch + 1) < max_epochs:
            continue

        regressor.eval()
        with torch.no_grad():
            va_pred = regressor(X_va.to(device, non_blocking=False))
            va_loss = loss_fn(va_pred, y_va.to(device, non_blocking=False)).item()

        if va_loss + 1e-9 < best_loss:
            best_loss = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in regressor.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.early_patience:
                break

    regressor.load_state_dict(best_state)
    regressor.eval().to("cpu")
    return regressor


def update_client_surrogate_rep(
    local_ID: int,
    local_images: np.ndarray,
    Z_query_negatives: np.ndarray,
    state_dict_buffer,
    regressor_buffer_rep,
    gpu_id: Optional[int],
    pe_bands: int = 4,
    seed: Optional[int] = None,
) -> nn.Module:
    # Load encoder -> embed local images -> build training pairs -> train regressor.
    device = get_device(gpu_id)
    set_torch_deterministic(seed)

    state_dict_buffer.seek(0)
    enc_state = torch.load(state_dict_buffer, map_location="cpu")

    encoder = FCNetwork_mnist(in_dim=784, feat_dim=2)
    encoder.load_state_dict(enc_state)
    encoder.to(device).eval()

    with torch.no_grad():
        imgs = _as_tensor(local_images, device=device)
        if imgs.ndim == 4 and imgs.shape[1] == 1:
            imgs = imgs.view(imgs.shape[0], -1)
        Z_local = encoder(imgs).detach().cpu().numpy()

    Z_query_negatives = np.asarray(Z_query_negatives, dtype=np.float32)

    X_neg, y_neg = build_surrogate_dataset_neg(Z_query_negatives, Z_local)

    cfg = TrainConfig(pe_bands=pe_bands, seed=seed)
    regressor = train_surrogate_rep(X_neg, y_neg, regressor_buffer=regressor_buffer_rep, device=device, cfg=cfg)
    return regressor
