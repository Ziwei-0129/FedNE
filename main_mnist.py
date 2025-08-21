from __future__ import annotations

import argparse
import copy
import io
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from numpy.random import beta as np_beta
from scipy.spatial.distance import pdist, squareform

from Dataset.mnist_datasets import get_mnist_dataset
from utils.misc_dir import create_folder, seed_everything
from utils.networks import FCNetwork_mnist
from utils.surrogate_finetune_optimized import update_client_surrogate_rep
# from utils.surrogate_finetune import update_client_surrogate_rep
from utils.utils_neigh import build_kNN_graph, find_kNN
from utils.utils_syn import sample_neighborhood_2D
from utils.utils_train import (
    FedAvg,
    clientUpdate,
    plt_global_wClientLabels,
    split_row_col,
    test_global,
)



# ----------------------------- Utilities -----------------------------

def get_device(gpu_id: int | None) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}" if gpu_id is not None else "cuda")
    return torch.device("cpu")


def to_tensor(x: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.as_tensor(x, device=device, dtype=dtype)


# -------------------------- Mixup + local kNN -------------------------

def intra_client_mixup(
    X: np.ndarray,
    closest_neighbors: np.ndarray,
    k: int,
    alpha: float = 0.2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Lightweight intra-client mixup.
    - X: (N, D)
    - closest_neighbors: (N, >=k) each row sorted by increasing distance
    - choose a random neighbor among the first k for each row
    - lam ~ Beta(alpha, alpha)
    """
    N, D = X.shape
    rng = rng or np.random.default_rng()
    # sample lambda per-sample, keep broadcasting-friendly shape
    lam = np.asarray(np_beta(alpha, alpha, N), dtype=np.float32).reshape(N, 1)

    # pick one neighbor index from [0, k-1]
    k = max(1, int(k))
    if closest_neighbors.shape[1] < k:
        raise ValueError(f"closest_neighbors has {closest_neighbors.shape[1]} cols < k={k}")
    sel = closest_neighbors[np.arange(N), rng.integers(low=0, high=k, size=N)]

    mixed = lam * X + (1.0 - lam) * X[sel]
    return mixed.astype(np.float32, copy=False)


def client_mixup_and_kNN_graph(
    client_sets: Dict[int, np.ndarray],
    closest_neighbors_set: Dict[int, np.ndarray],
    k: int,
    n_users: int,
    rng: np.random.Generator,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return (client_affinity_set, client_mixed_sets)."""
    client_affinity_set: Dict[int, np.ndarray] = {}
    client_mixed_sets: Dict[int, np.ndarray] = {}

    for c in range(n_users):
        X = client_sets[c]
        X_mix = intra_client_mixup(X, closest_neighbors_set[c], k=k, alpha=0.2, rng=rng)
        client_mixed_sets[c] = X_mix

        X_cat = np.vstack([X, X_mix])
        neigh_inds_self = find_kNN(X_cat, X_cat, k=k, type="loo")
        client_affinity_set[c] = neigh_inds_self

    return client_affinity_set, client_mixed_sets


# ----------------------------- Main logic -----------------------------

@dataclass
class Args:
    seed: int
    dataset: str
    k: int
    test_k: int
    rounds: int
    n_users: int
    iid: bool
    alpha: float
    n_data: int | None
    epochs_local: int
    batch_size: int
    n_batches: int
    surrogate: bool
    start_round: int
    test_bs: int
    lr: float
    path: str
    checkpoint: str | None
    step_size: float
    steps: int
    n_intervals: int
    beta: float
    pool_size: int
    gpu_ids: List[int] | None
    gpu_id: int
    folder_path: str | None = None


def build_closest_neighbors(client_sets: Dict[int, np.ndarray], k: int, n_users: int) -> Dict[int, np.ndarray]:
    """Compute per-client top-(k*10) nearest neighbor lists using Euclidean distance."""
    closest_neighbors_set: Dict[int, np.ndarray] = {}
    topk = max(1, int(k) * 10)
    for c in range(n_users):
        X = client_sets[c]
        # pairwise distances within client
        d = squareform(pdist(X, metric="euclidean"))
        np.fill_diagonal(d, np.inf)
        closest_neighbors_set[c] = np.argsort(d, axis=1)[:, :topk]
    return closest_neighbors_set


def main(args: Args):
    # Deterministic seed 
    seed_everything(args.seed)
    rng = np.random.default_rng(args.seed)

    # Create output folder and device
    folder_path = create_folder(args, isCent=False)
    args.folder_path = folder_path
    os.makedirs(os.path.join(folder_path, "saved_encoders"), exist_ok=True)

    device = get_device(args.gpu_id)

    # ---------------------------- Data & Model ---------------------------
    if "mnist" in args.dataset:
        images_train, labels_train, client_sets, client_labels, dict_users = get_mnist_dataset(
            args, args.dataset, folder_path, isCent=False
        )
        encoder = FCNetwork_mnist(in_dim=784, feat_dim=2)
    else:
        raise ValueError("Unknown dataset. Expected a MNIST variant.")

    # Optional preload
    if args.checkpoint is not None and os.path.isfile(args.checkpoint):
        encoder.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    # Global kNN graph (for test metric)
    _, graph_glob, _ = build_kNN_graph(dataset=images_train, n_nbrs=args.k)

    # Initial evaluation
    test_loss, test_loss_pos, test_loss_neg = test_global(
        copy.deepcopy(encoder).eval(), images_train, args.test_k, args.test_bs, args.seed, graph=graph_glob
    )
    print(f"Initial test loss: {test_loss:.6f}, {test_loss_pos:.6f}, {test_loss_neg:.6f}")

    # Client stats
    client_sizes = [client_sets[i].shape[0] for i in range(args.n_users)]
    n_total = int(np.sum(client_sizes))
    client_ratios = [sz / n_total for sz in client_sizes]
    print(f"client sizes: {client_sizes}\nTotal: {n_total}  ratios: {client_ratios}")

    # Precompute per-client neighbor lists
    closest_neighbors_set = build_closest_neighbors(client_sets, args.k, args.n_users)
    

    # ------------------------ Federated Training -------------------------
    lr_decay_rounds = [int(0.2 * args.rounds), int(0.3 * args.rounds), int(0.6 * args.rounds)]
    surrogate_start_round = int(np.floor(0.3 * args.rounds))
    
    txt_path = os.path.join(folder_path, "losses.txt")
    with open(txt_path, "w") as f:
        f.write("train_index  loss_sum  attractive_loss  repulsive_loss\n")

    nrows, ncols = split_row_col(args.n_users)

    global_weights = copy.deepcopy(encoder.state_dict())
    learning_rate = float(args.lr)
    batch_size = args.batch_size

    # Track per-client attraction thresholds if used downstream
    client_attraction_dict = {j: 0.0 for j in range(args.n_users)}

    # Surrogate regressor cache (persist across rounds)
    surrogate_losses_rep_dict: Dict[int, nn.Module | None] = {c: None for c in range(args.n_users)}

    for r in range(args.rounds):
        # Learning rate scheduling
        if r in lr_decay_rounds:
            learning_rate *= 0.1
            print(f"[Scheduler] Round {r}: decayed LR --> {learning_rate:.6g}")
            
        # Build mixup augments and per-client kNN on concatenated local + mixed
        rand = np.random.default_rng(int(time.time()))
        client_affinity_set, client_mixed_sets = client_mixup_and_kNN_graph(
            client_sets, closest_neighbors_set, int(args.k), args.n_users, rand #rng
        )

        # ----------------- Train repulsion surrogate (optional) -----------------
        if r >= surrogate_start_round:
            # Use the current global encoder frozen to generate z for neg sampling ranges
            encoder.load_state_dict(global_weights)
            encoder.to(device).eval()
            batch_size *= 2

            neg_samples_2d_dict: Dict[int, np.ndarray] = {}
            with torch.no_grad():
                for c in range(args.n_users):
                    X_local = client_sets[c]
                    Z_local = encoder(to_tensor(X_local, device)).cpu().numpy()

                    # bounding box center and size (fixed bug in original range_x/y)
                    min_x, max_x = Z_local[:, 0].min(), Z_local[:, 0].max()
                    min_y, max_y = Z_local[:, 1].min(), Z_local[:, 1].max()
                    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
                    range_x = max(cx - min_x, max_x - cx) * 2.0
                    range_y = max(cy - min_y, max_y - cy) * 2.0

                    neg_samples_2d_dict[c] = sample_neighborhood_2D(
                        center=np.array([cx, cy], dtype=np.float32),
                        range_x=float(range_x),
                        range_y=float(range_y),
                        step_size=float(args.step_size),
                    )

            # Train or fine-tune each client's surrogate regressor
            for client_ID in range(args.n_users):
                # Serialize encoder weights in CPU for the child util
                state_buf = io.BytesIO()
                torch.save({k: v.cpu() for k, v in encoder.state_dict().items()}, state_buf)

                regressor_buf = None
                if surrogate_losses_rep_dict[client_ID] is not None:
                    # If you later want incremental fine-tuning, you can pass a real buffer here
                    regressor_buf = None

                regressor_rep = update_client_surrogate_rep(
                    client_ID,
                    client_sets[client_ID],
                    neg_samples_2d_dict[client_ID],
                    state_buf,
                    regressor_buf,
                    args.gpu_id,
                )
                surrogate_losses_rep_dict[client_ID] = regressor_rep.cpu()

        # -------------------------- Client local updates ------------------------
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6, 6))
        axes = np.array(ax).reshape(-1) if isinstance(ax, np.ndarray) else np.array([ax])

        all_weights: List[dict] = []

        for cnt in range(args.n_users):
            print(f"\n------ Local training Client {cnt + 1}/{args.n_users}, Round {r} ------")
            local_data = client_sets[cnt]
            local_labels = client_labels[cnt]

            # Concat original + mixup for this client's local training epoch(s)
            clientdata_mixed = np.vstack([local_data, client_mixed_sets[cnt]])

            embedder, z_umap, mean_att_loss = clientUpdate(
                isSurrogate=args.surrogate,
                client_ratios=client_ratios,
                client_id=cnt,
                local_data=clientdata_mixed,
                local_labels=local_labels,
                n_clients=args.n_users,
                encoder=copy.deepcopy(encoder).cpu(),
                k=args.k,
                lr=learning_rate,
                epochs_local=args.epochs_local,
                batch_size=batch_size,
                n_batches=args.n_batches,
                client_graph_info=client_affinity_set[cnt],
                client_funct_dict=[None, surrogate_losses_rep_dict],
                isCent=False,
                add_both=False,
                client_attraction_thred=client_attraction_dict[cnt],
            )

            # Collect *only* the encoder (model[1]) weights
            w_local = copy.deepcopy(embedder.model[1]).cpu().state_dict()
            all_weights.append(w_local)

            # Plot client embeddings (only original points)
            ax_i = axes[cnt]
            n_local = local_data.shape[0]
            z_local_only = z_umap[:n_local]
            if labels_train is None:
                colors, cmap = "gray", None
            else:
                colors = np.array(local_labels, dtype=int)
                cmap = "tab10" if args.n_users == 10 else "Set1"
            ax_i.scatter(z_local_only[:, 0], z_local_only[:, 1], c=colors, cmap=cmap, s=5, alpha=1, rasterized=True)
            ax_i.axis("equal")

        plt.tight_layout()
        plt.close(fig)

        # --------------------------- FedAvg + Save ----------------------------
        global_weights = FedAvg(all_weights, ratios=client_ratios)
        encoder.load_state_dict(global_weights)

        # Save checkpoints regularly and at the end
        ckpt_dir = os.path.join(folder_path, "saved_encoders")
        torch.save(encoder.state_dict(), os.path.join(ckpt_dir, "encoder_current.p"))

        # ----------------------- Global projection & test ----------------------
        encoder_glob = copy.deepcopy(encoder).eval().to(device)
        with torch.no_grad():
            _ = encoder_glob(to_tensor(images_train, device))

        test_loss, test_loss_pos, test_loss_neg = test_global(
            copy.deepcopy(encoder).eval().cpu(), images_train, args.test_k, args.test_bs, args.seed, graph=graph_glob
        )

        # Append to log file
        with open(os.path.join(folder_path, "losses.txt"), "a") as f:
            f.write(f"R{r} {test_loss} {test_loss_pos} {test_loss_neg}\n")


# ------------------------------ CLI wrapper ------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dataset", type=str, required=True, help="e.g., 'mnist' or 'mnist-iid'")
    parser.add_argument("--k", type=int, required=True, help="k for k-NN")
    parser.add_argument("--test_k", type=int, default=7)

    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--n_users", type=int, required=True)
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_data", type=int, default=None)

    # Local training
    parser.add_argument("--epochs_local", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--n_batches", type=int, default=1)

    # Surrogate
    parser.add_argument("--surrogate", action="store_true")
    parser.add_argument("--start_round", type=int, default=-1)

    # Testing
    parser.add_argument("--test_bs", type=int, default=1000)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--path", type=str, default=os.path.join("Results_mixup", "mnist"))

    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--step_size", type=float, default=0.3)
    parser.add_argument("--steps", type=int, default=2)

    parser.add_argument("--n_intervals", type=int, default=3)
    parser.add_argument("--beta", type=float, default=0.2)

    parser.add_argument("--pool_size", type=int, default=10)
    parser.add_argument("--gpu_ids", nargs="+", type=int)
    parser.add_argument("--gpu_id", type=int, default=0)

    cli = parser.parse_args()

    # ensure output root exists
    os.makedirs(cli.path, exist_ok=True)

    main(Args(**vars(cli)))