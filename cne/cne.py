import copy
import torch
import time
import sys
import os
import numpy as np

from cne.cne_utils import criterion_client_pos, criterion_client_neg, generate_batch, \
    criterion_attraction, criterion_repulsion, make_neg_inds

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class FastTensorDataLoader:
    def __init__(self, n_local_negatives, indices_local, indices_neighs,
                 batch_size=-1, n_batches=1, shuffle=True, on_gpu=True, drop_last=False, seed=None):
        # Flatten (item, neigh) pairs
        k = len(indices_neighs[0])
        n_local = len(indices_local)
        n_rows = n_local * k

        # repeat each local idx k times, flatten neighbors once
        indices_local_dups = np.repeat(indices_local, k).astype(np.int64)
        indices_neighs_flat = indices_neighs.ravel().astype(np.int64)

        # Keep ints in Python space for indexing/slicing
        self.batch_size_items = (batch_size if batch_size > 0 else n_local) * k
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.k = k
        self.n_rows = n_rows

        # RNG (deterministic if seed passed)
        self.gen = torch.Generator(device="cuda" if (on_gpu and torch.cuda.is_available()) else "cpu")
        if seed is not None: self.gen.manual_seed(seed)

        # Pre-load to chosen device once
        self.device = torch.device("cuda") if (on_gpu and torch.cuda.is_available()) else torch.device("cpu")
        self.items = torch.as_tensor(indices_local_dups, device=self.device, dtype=torch.long)
        self.neighs = torch.as_tensor(indices_neighs_flat, device=self.device, dtype=torch.long)

        # Negatives are generated lazily per batch (see __next__)
        self.m = 5
        self.n_local = n_local

        # number of batches
        nb, rem = divmod(self.n_rows, self.batch_size_items)
        self.n_batches = nb + (1 if rem > 0 and not self.drop_last else 0)

    def __iter__(self):
        if self.shuffle:
            self.perm = torch.randperm(self.n_rows, generator=self.gen, device=self.device)
        else:
            self.perm = None
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.n_rows:
            raise StopIteration
        j = min(self.i + self.batch_size_items, self.n_rows)
        if self.perm is None:
            idx = slice(self.i, j)
            items = self.items[idx]
            neighs = self.neighs[idx]
        else:
            take = self.perm[self.i:j]
            items = self.items.index_select(0, take)
            neighs = self.neighs.index_select(0, take)

        # negatives: sample m per row from [0, n_local) on the SAME device
        neg = torch.randint(0, self.n_local, (items.numel(), self.m), generator=self.gen, device=self.device)

        self.i = j
        return items, neighs, neg

    def __len__(self):
        return self.n_batches




########################################################
#                 Test function
########################################################
def test(
    test_loader,
    model,
    log_Z,
    criterion,
    force_resample=None,
):
    model.eval()
    losses = []
    losses_pos, losses_neg = [], []

    with torch.no_grad():
        for idx, (item, neigh) in enumerate(test_loader):
            images = torch.cat([item, neigh], dim=0)
            images = images.to(next(model.parameters()).device)

            # compute loss
            features = model(images)
            force_resample = force_resample if force_resample is not None else idx == 0
            loss, loss_pos, loss_neg = criterion(features, log_Z, force_resample=force_resample, isTest=True)

            # update metric
            losses.append(loss.item())
            losses_pos.append(loss_pos.item())
            losses_neg.append(loss_neg.item())

    return losses, losses_pos, losses_neg





########################################################
#             Train function: surrogate
########################################################

def train_surrogate(
    n_local_data,
    model,
    log_Z,
    optimizer,
    client_id=-1,
    client_ratios=None,
    n_batches=1,
    batch_size=-1,
    indices_att_self=None,
    surrogates_att=None,
    surrogates_rep=None,
    client_attraction_thred=None,
    add_both=False,
    clip_grad=True,
):

    model.train()
    losses = []

    n_local = model[0].num_embeddings


    " Generate batched data "
    indices_local = np.arange(n_local, dtype=np.int64)
    
    n_local_negatives = n_local_data

    train_loader = FastTensorDataLoader(n_local_negatives, indices_local, indices_neighs=indices_att_self,
                                        shuffle=True, batch_size=batch_size, n_batches=n_batches, on_gpu=True)



    #####################################################
    #               Compute att/repl losses
    #####################################################
    for idx, (item, neigh, negatives) in enumerate(train_loader):

        if n_batches > 0 and idx >= n_batches:
            break


        " *** Using Others' Surrogate Functions *** "
        if surrogates_rep is not None:
            
            
            feature_item = model(item)
            feature_item = positional_encoding(feature_item, L=4) 

            " LOCAL losses "
            loss_att = criterion_attraction(model, item, neigh, loss_mode=None)
            loss_rep = criterion_repulsion(model, item, negatives, loss_mode=None)
            
            
            sum_loss_p = loss_att
            sum_loss_n = loss_rep * client_ratios[client_id]
 
            for key in surrogates_rep:
                if key == client_id:
                    continue
                
                if surrogates_rep[key] is None:
                    continue

                client_func_ci_rep = surrogates_rep[key].to(device).eval()
                surrogate_losses_rep = client_func_ci_rep(feature_item)[:, 0]
 
                sum_loss_n += (surrogate_losses_rep * client_ratios[key] * 1.0)


            sum_loss_p = sum_loss_p.mean()
            sum_loss_n = sum_loss_n.mean() * 5.0


        else:
            " client LOCAL losses "
            loss_att = criterion_attraction(model, item, neigh, loss_mode=None)
            loss_rep = criterion_repulsion(model, item, negatives, loss_mode=None)

            sum_loss_p = loss_att.mean()
            sum_loss_n = loss_rep.mean() * 5.0


        # Sum up
        loss = sum_loss_p + sum_loss_n

        if client_attraction_thred == 0.0:
            losses.append(sum_loss_p.item())
        else:
            losses.append(loss_att.mean().item())
            

        # Update parameters
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if clip_grad:
            torch.nn.utils.clip_grad_value_(model.parameters(), 4)
            # if log_Z is not None:
            #     torch.nn.utils.clip_grad_value_(log_Z, 4)

        optimizer.step()

    return losses




def positional_encoding(x: torch.Tensor, L: int = 4) -> torch.Tensor:
    if L <= 0:
        return x
    freqs = (2.0 ** torch.arange(L, device=x.device, dtype=x.dtype)).view(1, L, 1)
    angles = x.unsqueeze(1) * freqs
    return torch.cat([x, torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)], dim=1)




###################################################################################
#     Class for computing contrastive embeddings from similarity information
###################################################################################
class ContrastiveEmbedding(object):

    def __init__(
        self,
        model: torch.nn.Module,
        negative_samples=5,
        n_epochs=50,
        n_batches=None,
        batch_size=-1,
        client_ratios=None,
        device="cuda:0",
        learning_rate=0.001,
        lr_min_factor=0.1,
        momentum=0.9,
        temperature=0.5,
        noise_in_estimator=1.0,
        Z_bar=None,
        s=None,
        eps=1.0,
        clamp_high=1.0,
        clamp_low=1e-4,
        Z=1.0,
        loss_mode="umap",
        metric="euclidean",
        optimizer="adam",
        weight_decay=0,
        anneal_lr="none",
        lr_decay_rate=0.1,
        lr_decay_epochs=None,  # unused for now
        clip_grad=True,
        save_freq=25,
        callback=None,
        print_freq_epoch=None,
        print_freq_in_epoch=None,
        seed=0,
        loss_aggregation="sum", #"mean",
        force_resample=None,
        warmup_epochs=0,
        warmup_lr=0,
        n_clients=-1,
        client_funct_dict=None,
        client_id=-1,
    ):
        """
        :param model: torch.nn.Module Embedding model (embedding layer for non-parametric, neural network for parametric)
        :param batch_size: int Batch size
        :param negative_samples: int Number of negative samples per positive sample
        :param n_epochs: int Number of optimization epochs
        :param device: torch.device Device of optimization
        :param learning_rate: float Learning rate
        :param lr_min_factor: float Minimal value to which learning rate is annealed
        :param momentum: float Momentum of SGD
        :param temperature: float Temperature used in Cosine similarity
        :param noise_in_estimator: float Value used in negative sampling's fraction q / (q+ noise_in_estimator), redundant with Z_bar
        :param Z_bar: float Fixed normalization constant in negative sampling, redundant with noise_in_estimator
        :param s: float Slider parameter setting the fixed normalization constant, redundant with noise_in_estimator
        :param eps: float Iterpolates between UMAP's implicit similarity (eps = 0) and the Cauchy kernels (eps = 1.0)
        :param clamp_high: float Upper value at which arguments to logarithms are clamped.
        :param clamp_low: float Lower value at which arguments to logarithms are clamped.
        :param Z: float Initial value for the learned normalization parameter of NCE
        :param loss_mode: str Specifies which loss to use. Must be one of "umap", "neg", "nce", "infonce", "infonce_alt". "neg_sample" is depricated and defaults to "neg"
        :param metric: str Specifies which metric to use for computing distances. Must be "cosine" or "euclidean".
        :param optimizer: str Specifies which optimizer to use. Must be "sgd" or "adam"
        :param weight_decay: float Value of weight decay.
        :param anneal_lr: bool If True, the learning rate is annealed
        :param lr_decay_rate: float Parameter for speed of learing rate decay
        :param lr_decay_epochs: int Number of epochs over which learning rate is decayed
        :param clip_grad: bool If True, gradients are clipped
        :param save_freq: int Frequency in epochs of calling callback.
        :param callback: callable Callback to call before first and every save_freq epochs.
        :param print_freq_epoch: int Epoch progress is printed every print_freq_epoch epoch
        :param print_freq_in_epoch: int Losses are printed every print_freq_in_epoch batch per epoch
        :param seed: int Random seed
        :param loss_aggregation: str Specifies how to aggregate loss over a batch. Must be "sum" or "mean".
        :param force_resample: bool or None If True, negative sample indices are resampled every batch. If None, they are resampled every epoch.
        :param warmup_epochs: int Number of epochs for linearly warming up the learning rate
        :param warmup_lr: float Starting learning rate to warm up from.
        """
        self.model: torch.nn.Module = model
        self.batch_size: int = batch_size
        self.n_batches = n_batches
        self.client_ratios: list = client_ratios
        self.negative_samples: int = negative_samples
        self.n_epochs: int = n_epochs
        self.device = device
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.temperature = temperature
        self.loss_mode: str = loss_mode
        self.metric: str = metric
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        if isinstance(anneal_lr, bool):
            anneal_lr = "linear" if anneal_lr else "none"
        self.anneal_lr: str = anneal_lr
        self.lr_min_factor: float = lr_min_factor
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_epochs = lr_decay_epochs
        self.warmup_lr = warmup_lr
        self.warmup_epochs = warmup_epochs
        self.clip_grad: bool = clip_grad
        self.save_freq: int = save_freq
        self.callback = callback
        self.print_freq_epoch = print_freq_epoch
        self.print_freq_in_epoch = print_freq_in_epoch
        self.eps = eps
        self.clamp_high = clamp_high
        self.clamp_low = clamp_low
        self.seed = seed
        self.loss_aggregation = loss_aggregation
        self.force_resample = force_resample
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.client_funct_dict = client_funct_dict
        self.n_clients = n_clients
        self.client_id = client_id
        self.log_Z = torch.tensor(np.log(Z), device=self.device)

        # alias for loss mode "neg" to ensure backwards compatibility
        # since the loss mode is put into the file names, which are featured in the notebooks,
        # we keep "neg_sample" internally
        if self.loss_mode == "neg":
            self.loss_mode = "neg_sample"

        if self.loss_mode == "nce":
            self.log_Z = torch.nn.Parameter(self.log_Z, requires_grad=True)

        if self.loss_mode == "neg_sample":
            n_specified_params = (noise_in_estimator is not None) + (Z_bar is not None) + (s is not None)
            assert (
                n_specified_params > 0
                # noise_in_estimator is not None or Z_bar is not None or s is not None
            ), f"Exactly one of 'noise_in_estimator', 'Z_bar' and 's' must be not None."

            if n_specified_params > 1:
                print(
                    "Warning: More than one of 'noise_in_estimator', 'Z_bar' and "
                    "'s' were specified. 's' will supersede 'Z_bar', which supersedes 'noise_in_estimator'."
                )
        self.s = s
        self.Z_bar = Z_bar
        self.noise_in_estimator = noise_in_estimator

        # move to correct device at init, esp before registering with the optimizer
        self.model = self.model.to(self.device)




    def fit(
        self,
        X,
        Y,
        isSurrogate=None,
        client_graph_info=None,
        client_attraction_thred=None,
        isCent=False,
        add_both=True,
        gpu_id=0,
    ):

        global device
        device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device


        n_local_data = len(Y)

        # set up optimizer
        params = [{"params": self.model.parameters()}]
        if self.loss_mode == "nce":
            params += [
                {"params": self.log_Z, "lr": 0.001}
            ]  # make sure log_Z always has a sufficiently small lr


        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params,
                weight_decay=self.weight_decay,
                lr=self.learning_rate,
            )
        else:
            raise ValueError(
                f"Only optimizer 'adam' and 'sgd' allowed, but is {self.optimizer}."
            )

        # initial callback
        if (
            self.save_freq is not None
            and self.save_freq > 0
            and callable(self.callback)
        ):
            self.callback(
                -1,
                self.model,
                self.negative_samples,
                self.loss_mode,
                self.log_Z,
            )

        batch_losses = []

        # logging memory usage
        mem_dict = {
            "active_bytes.all.peak": [],
            "allocated_bytes.all.peak": [],
            "reserved_bytes.all.peak": [],
            "reserved_bytes.all.allocated": [],
        }


        ############################################################
        #                    train for epochs
        ############################################################
        for epoch in range(self.n_epochs):
            # if "cuda" in self.device:
            #     info = torch.cuda.memory_stats(self.device)
            #     [mem_dict[k].append(info[k]) for k in mem_dict.keys()]

            # anneal learning rate
            lr = new_lr(
                self.learning_rate,
                self.anneal_lr,
                self.lr_decay_rate,
                lr_min_factor=self.lr_min_factor,
                cur_epoch=epoch,
                total_epochs=self.n_epochs,
                decay_epochs=self.lr_decay_epochs,
                warmup_epochs=self.warmup_epochs,
                warmup_lr=self.warmup_lr,
            )
            print(f'anneal learning rate = {lr}')

            # just change the lr of the first param group, not that of Z
            optimizer.param_groups[0]["lr"] = lr


            ################################################################
            ########################### Training ###########################

            if isSurrogate:
                bl = train_surrogate(
                        # local_labels=Y,
                        n_local_data=n_local_data,
                        model=self.model,
                        log_Z=self.log_Z,
                        optimizer=optimizer,
                        client_id=self.client_id,
                        client_ratios=self.client_ratios,
                        n_batches=self.n_batches,
                        batch_size=self.batch_size,
                        indices_att_self=client_graph_info,
                        surrogates_att=self.client_funct_dict[0],
                        surrogates_rep=self.client_funct_dict[1],
                        client_attraction_thred=client_attraction_thred,
                        add_both=add_both,
                )

                batch_losses.append(bl)


            # callback
            if (
                self.save_freq is not None
                and self.save_freq > 0
                and epoch % self.save_freq == 0
                and callable(self.callback)
            ):
                self.callback(
                    epoch, self.model, self.negative_samples, self.loss_mode, self.log_Z
                )
            # print epoch progress
            if self.print_freq_epoch is not None and epoch % self.print_freq_epoch == 0:
                print(f"Finished epoch {epoch}/{self.n_epochs}   loss {np.mean(batch_losses)}", file=sys.stderr)

        self.losses = batch_losses
        self.mem_dict = mem_dict
        self.embedding_ = None
        return self



    def fit_transform(self, X):
        self.fit(X)
        return self.embedding_




    ######################################################################
    #                       Test on global dataset
    ######################################################################
    def compute_test_loss(self, X: torch.utils.data.DataLoader, loss_aggregation='mean'):
        # set up loss
        criterion = ContrastiveLoss(
            negative_samples=self.negative_samples,
            metric=self.metric,
            temperature=self.temperature,
            loss_mode=self.loss_mode,
            noise_in_estimator=torch.tensor(self.noise_in_estimator).to(self.device),
            eps=torch.tensor(self.eps).to(self.device),
            clamp_high=self.clamp_high,
            clamp_low=self.clamp_low,
            seed=self.seed,
            loss_aggregation=loss_aggregation,
            lossSplit=True,)

        # test
        batch_losses, batch_losses_pos, batch_losses_neg = test(X, self.model, self.log_Z, criterion, force_resample=self.force_resample,)

        self.loss = np.mean(batch_losses)
        self.embedding_ = None
        return self, self.loss, np.mean(batch_losses_pos), np.mean(batch_losses_neg)




######################################################################
#                       Contrastive NE Loss
######################################################################
class ContrastiveLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self,
        negative_samples=5,
        temperature=0.07,
        loss_mode="all",
        metric="euclidean",
        base_temperature=1,
        eps=1.0,
        noise_in_estimator=1.0,
        clamp_high=1.0,
        clamp_low=1e-4,
        seed=0,
        loss_aggregation="mean",
        lossSplit=False,
    ):
        super(ContrastiveLoss, self).__init__()
        self.negative_samples = negative_samples
        self.temperature = temperature
        self.loss_mode = loss_mode
        self.metric = metric
        self.base_temperature = base_temperature
        self.noise_in_estimator = noise_in_estimator
        self.eps = eps
        self.clamp_high = clamp_high
        self.clamp_low = clamp_low
        self.seed = seed
        torch.manual_seed(self.seed)
        self.neigh_inds = None
        self.loss_aggregation = loss_aggregation
        self.lossSplit = lossSplit


    def forward(self, features, log_Z=None, force_resample=False, mode=None, neighbors=None, isTest=False):
        """Compute loss for model. SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [2 * bsz, n_views, ...].
            log_Z: scalar, logarithm of the learnt normalization constant for nce.
            force_resample: Whether the negative samples should be forcefully resampled.
        Returns:
            A loss scalar.
        """

        if isTest:
            device = torch.device('cuda:0')


        batch_size = features.shape[0] // 2
        b = batch_size

        # We can at most sample this many samples from the batch.
        # `b` can be lower than `self.negative_samples` in the last batch.
        negative_samples = min(self.negative_samples, 2 * b - 1)

        if neighbors is None:
            if force_resample or self.neigh_inds is None:
                neigh_inds = make_neighbor_indices(
                    batch_size, negative_samples, device=features.device
                )
                self.neigh_inds = neigh_inds

            else:
                neigh_inds = self.neigh_inds

            neighbors = features[neigh_inds]


        # `neigh_mask` indicates which samples feel attractive force
        # and which ones repel each other
        # neigh_mask = torch.ones_like(neigh_inds, dtype=torch.bool)
        neigh_mask = torch.ones((b, 6), dtype=torch.bool).to(device)
        neigh_mask[:, 0] = False

        origs = features[:b]

        # compute probits
        if self.metric == "euclidean":
            dists = ((origs[:, None] - neighbors) ** 2).sum(axis=2)
            # Cauchy affinities
            probits = torch.div(1, self.eps + dists)

        elif self.metric == "cosine":
            norm = torch.nn.functional.normalize
            o = norm(origs).unsqueeze(1)
            n = norm(neighbors).transpose(1, 2)
            logits = torch.bmm(o, n).squeeze() / self.temperature
            probits = torch.exp(logits)
        else:
            raise ValueError(f"Unknown metric “{self.metric}”")


        # compute loss
        if self.loss_mode == "nce":
            # for proper nce it should be negative_samples * p_noise. But for
            # uniform noise distribution we would need the size of the dataset
            # here. Also, we do not use a uniform noise distribution as we sample
            # negative samples from the batch.

            if self.metric == "euclidean":
                # estimator is (cauchy / Z) / ( cauchy / Z + neg samples)). For numerical
                # stability rewrite to 1 / ( 1 + (d**2 + eps) * Z * m)
                estimator = 1 / (
                    1 + (dists + self.eps) * torch.exp(log_Z) * negative_samples
                )
            else:
                probits = probits / torch.exp(log_Z)
                estimator = probits / (probits + negative_samples)

            loss = -(~neigh_mask * torch.log(estimator.clamp(self.clamp_low, self.clamp_high))) - (
                neigh_mask * torch.log((1 - estimator).clamp(self.clamp_low, self.clamp_high))
            )

        elif self.loss_mode == "neg_sample":
            if self.metric == "euclidean":
                # estimator rewritten for numerical stability as for nce
                estimator = 1 / (1 + self.noise_in_estimator * (dists + self.eps))
                # estimator = torch.div(1, self.eps + dists)

            else:
                estimator = probits / (probits + self.noise_in_estimator)

            loss_pos = -(~neigh_mask * torch.log(estimator.clamp(self.clamp_low, self.clamp_high)))
            loss_neg = -(neigh_mask * torch.log((1 - estimator).clamp(self.clamp_low, self.clamp_high)))

            loss = -(~neigh_mask * torch.log(estimator.clamp(self.clamp_low, self.clamp_high))) - (
                neigh_mask * torch.log((1 - estimator).clamp(self.clamp_low, self.clamp_high))
            )  # !!!!!!!!!!!!!!!


        elif self.loss_mode == "umap":
            # cross entropy parametric umap loss
            loss = -(~neigh_mask * torch.log(probits.clamp(self.clamp_low, self.clamp_high))) - (
                neigh_mask * torch.log((1 - probits).clamp(self.clamp_low, self.clamp_high))
            )
        elif self.loss_mode == "infonce":
            # loss from e.g. sohn et al 2016, includes pos similarity in denominator
            loss = -(self.temperature / self.base_temperature) * (
                (torch.log(probits.clamp(self.clamp_low, self.clamp_high)[~neigh_mask]))
                - torch.log(probits.clamp(self.clamp_low, self.clamp_high).sum(axis=1))
            )
        elif self.loss_mode == "infonce_alt":
            # loss simclr
            loss = -(self.temperature / self.base_temperature) * (
                (torch.log(probits.clamp(self.clamp_low, self.clamp_high)[~neigh_mask]))
                - torch.log((neigh_mask * probits.clamp(self.clamp_low, self.clamp_high)).sum(axis=1))
            )
        else:
            raise ValueError(f"Unknown loss_mode “{self.loss_mode}”")


        if self.lossSplit and mode == "original":
            loss_pos = torch.mean(loss_pos, dim=1)
            loss_neg = torch.mean(loss_neg, dim=1)
            return loss, loss_pos, loss_neg

        if self.lossSplit:
            if self.loss_aggregation == "sum":
                loss_pos = loss_pos[:,0].sum()
                loss_neg = loss_neg[:,1:].sum()
                loss = loss.sum()
            elif self.loss_aggregation == "original":
                loss_pos = loss_pos[:,0]
                loss_neg = loss_neg[:,1:]
                loss = loss
            else:
                loss_pos = loss_pos.mean()
                loss_neg = loss_neg.mean()
                loss = loss.mean()
            return loss, loss_pos, loss_neg

        # aggregate loss over batch
        if self.loss_aggregation == "sum":
            loss = loss.sum()
        elif self.loss_aggregation == "original":
            loss = loss
        else:
            loss = loss.mean()

        return loss




##################################################################
#                        Helper functions
##################################################################
def new_lr(
    learning_rate,
    anneal_lr,
    lr_decay_rate,
    lr_min_factor,
    cur_epoch,
    total_epochs,
    decay_epochs=None,  # unused for now
    warmup_lr=0,
    warmup_epochs=0,
):
    """
    Decays the learning rate
    :param learning_rate: float Current learning rate
    :param anneal_lr: str Specifies the learning rate annealing. Must be one of "none", "linear" or "cosine"
    :param lr_decay_rate: float Rate of cosine decay.
    :param lr_min_factor: float Minimal learning rate of linear decay.
    :param cur_epoch: int Current epoch
    :param total_epochs: int Total number of epochs
    :param decay_epochs: int Number of decay epochs (unused)
    :param warmup_epochs: int Number of epochs for linearly warming up the learning rate
    :param warmup_lr: float Starting learning rate to warm up from.
    :return: float New learning rate
    """
    anneal_epochs = total_epochs - warmup_epochs
    if cur_epoch < warmup_epochs:
        lr = warmup_lr + (learning_rate - warmup_lr) * cur_epoch / warmup_epochs
    else:
        cur_epoch = cur_epoch - warmup_epochs
        if anneal_lr == "none":
            lr = learning_rate
        elif anneal_lr == "linear":
            lr = learning_rate * max(lr_min_factor, 1 - cur_epoch / anneal_epochs)
        elif anneal_lr == "cosine":
            eta_min = 0
            lr = (
                eta_min
                + (learning_rate - eta_min)
                * (1 + np.cos(np.pi * cur_epoch / anneal_epochs))
                / 2
            )
        else:
            raise RuntimeError(f"Unknown learning rate annealing “{anneal_lr = }”")

    return lr




def make_neighbor_indices(batch_size, negative_samples, device=None):
    """
    Selects neighbor indices
    :param batch_size: int Batch size
    :param negative_samples: int Number of negative samples
    :param device: torch.device Device of the model
    :return: torch.tensor Neighbor indices
    :rtype:
    """
    b = batch_size

    if negative_samples < 2 * b - 1:
        # uniform probability for all points in the minibatch,
        # we sample points for repulsion randomly (batch x 5)
        neg_inds = torch.randint(0, 2 * b - 1, (b, negative_samples), device=device)
        neg_inds += (torch.arange(1, b + 1, device=device) - 2 * b)[:, None]

    else:
        # full batch repulsion
        all_inds1 = torch.repeat_interleave(
            torch.arange(b, device=device)[None, :], b, dim=0
        )
        not_self = ~torch.eye(b, dtype=bool, device=device)
        neg_inds1 = all_inds1[not_self].reshape(b, b - 1)

        all_inds2 = torch.repeat_interleave(
            torch.arange(b, 2 * b, device=device)[None, :], b, dim=0
        )
        neg_inds2 = all_inds2[not_self].reshape(b, b - 1)
        neg_inds = torch.hstack((neg_inds1, neg_inds2))

    # now add transformed explicitly
    neigh_inds = torch.hstack(
        (torch.arange(b, 2 * b, device=device)[:, None], neg_inds)
    )

    return neigh_inds
