import torch
import numpy as np

from cne import ContrastiveEmbedding
import time




# various datasets / dataloaders
class NeighborTransformData(torch.utils.data.Dataset):
    """Returns a pair of neighboring points in the dataset."""
    def __init__(
            self, dataset, neighbor_mat, random_state=None
    ):
        self.dataset = torch.tensor(dataset, dtype=torch.float32)
        self.neighbor_mat = neighbor_mat
        self.rng = np.random.default_rng(random_state)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        neighs = self.neighbor_mat[i].nonzero()[1]
        nidx = self.rng.choice(neighs)

        item = self.dataset[i]
        neigh = self.dataset[nidx]
        return item, neigh


class NeighborTransformIndices(torch.utils.data.Dataset):
    """Returns a pair of indices of neighboring points in the dataset."""
    def __init__(
            self, neighbor_mat, random_state=None
    ):
        neighbor_mat = neighbor_mat.tocoo()
        self.heads = torch.tensor(neighbor_mat.row)
        self.tails = torch.tensor(neighbor_mat.col)

    def __len__(self):
        return len(self.heads)

    def __getitem__(self, i):
        return self.heads[i], self.tails[i]



class NumpyToTensorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, reshape=None):
        self.dataset = torch.tensor(dataset, dtype=torch.float32)
        if reshape is not None:
            self.reshape = lambda x: np.reshape(x, reshape)
        else:
            self.reshape = lambda x: x

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item = self.dataset[i]
        return self.reshape(item)


class NumpyToIndicesDataset(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i


# based on https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/27014/6
class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    """
    def __init__(self, neighbor_mat, batch_size=1024, shuffle=False, on_gpu=False, gpu_id=0, drop_last=False, seed=0):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.
        :param on_gpu: If True, the dataset is loaded on GPU as a whole.
        :param drop_last: Drop the last batch if it is smaller than the others.
        :param seed: Random seed

        :returns: A FastTensorDataLoader.
        """

        neighbor_mat = neighbor_mat.tocoo()
        tensors = [torch.tensor(neighbor_mat.row),
                   torch.tensor(neighbor_mat.col)]
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)

        # manage device
        self.device = "cpu"
        if on_gpu:
            # self.device = "cuda"
            self.device = f"cuda:{gpu_id}"
            tensors = [tensor.to(self.device) for tensor in tensors]
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        torch.manual_seed(self.seed)

        # Calculate number of batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0 and not self.drop_last:
            n_batches += 1
        self.n_batches = n_batches

        self.batch_size = torch.tensor(self.batch_size, dtype=int).to(self.device)

    def __iter__(self):
        if self.shuffle:
            '''generate random seed'''
            torch.manual_seed(int(time.time()))
            self.indices = torch.randperm(self.dataset_len, device=self.device)

        else:
            self.indices = None
            torch.manual_seed(42)
            self.indices = torch.randperm(self.dataset_len, device=self.device)
        self.i = 0
        return self

    def __next__(self):
        if self.i > self.dataset_len - self.batch_size:
            raise StopIteration
        if self.indices is not None:
            indices = self.indices[self.i:self.i+self.batch_size]
            batch = tuple(torch.index_select(t, 0, indices) for t in self.tensors)
        else:
            batch = tuple(t[self.i:self.i+self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches



class CNE(object):
    """
    Manages contrastive neighbor embeddings.
    """
    def __init__(self,
                 CF_set={0: {0: None}},
                 n_clients=-1,
                 model=None,
                 k=-1,
                 parametric=True,
                 n_epochs=1,
                 n_batches=1,
                 batch_size=-1,
                 on_gpu=True,
                 seed=0,
                 loss_aggregation="mean",
                 anneal_lr=True,
                 learning_rate=0.001,
                 client_id=-1,
                 gpu_id=0,
                 **kwargs):
        """
        :param model: Embedding model
        :param k: int Number of nearest neighbors
        :param parametric: bool If True and model=None uses a parametric embedding model
        :param on_gpu: bool Load whole dataset to GPU
        :param seed: int Random seed
        :param loss_aggregation: str If 'mean' uses mean aggregation of loss over batch, if 'sum' uses sum.
        :param anneal_lr: bool If True anneal the learning rate linearly.
        :param kwargs:
        """
        self.CF_set = CF_set
        self.n_clients = n_clients
        self.learning_rate = learning_rate
        self.model = model
        self.k = k
        self.parametric = parametric
        self.on_gpu = on_gpu
        self.kwargs = kwargs
        self.seed = seed
        self.n_epochs = n_epochs
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.loss_aggregation = loss_aggregation
        self.anneal_lr = anneal_lr
        self.client_id = client_id
        self.gpu_id = gpu_id




    "Transform a dataset using the fitted model."
    def transform(self, X):
        if self.parametric:
            size = X.shape[0]
            X = X.reshape(X.shape[0], -1)
            self.dataset_plain = NumpyToTensorDataset(X)
            self.dl_unshuf = torch.utils.data.DataLoader(
                self.dataset_plain,
                shuffle=False,
                batch_size=size, #self.cne.batch_size,
            )
            device = f"cuda:{self.gpu_id}"
            model = self.model[1].to(device)
            embd = np.vstack([model(batch.to(device)).detach().cpu().numpy() for batch in self.dl_unshuf])
        else:
            embd = self.model.weight.detach().cpu().numpy()
        return embd




    " Fit the model, then transform. "
    def fit_transform(self, X, Y, isSurrogate=None,
                    client_graph_info=None, client_ratios=None, client_funct_dict=None,
                    client_attraction_thred=None, isCent=False, add_both=True):

        __self = self.fit(
                X,
                Y,
                isSurrogate=isSurrogate,
                client_graph_info=client_graph_info,
                client_ratios=client_ratios,
                client_funct_dict=client_funct_dict,
                client_attraction_thred=client_attraction_thred,
                isCent=isCent,
                add_both=add_both,
        )
         
        return self.transform(X), np.mean(__self.cne.losses)




    def fit_transform_ft(self, X, graph=None):

        _ = self.fit_ft(X, graph=graph)
        return self.model[1], self.transform(X)




    def fit(self, X, Y, isSurrogate=None, 
                client_graph_info=None, client_ratios=None, client_funct_dict=None,
                client_attraction_thred=None, isCent=False, add_both=True):

        # Load embedding engine
        self.cne = ContrastiveEmbedding(
                                negative_samples=5,
                                model=self.model,
                                n_epochs=self.n_epochs,
                                n_batches=self.n_batches,
                                batch_size=self.batch_size,
                                client_ratios=client_ratios,
                                seed=self.seed,
                                loss_aggregation=self.loss_aggregation,
                                anneal_lr=self.anneal_lr,
                                client_funct_dict=client_funct_dict,
                                n_clients=self.n_clients,
                                learning_rate=self.learning_rate,
                                client_id=self.client_id,
                                **self.kwargs)

        # fit the model
        _ = self.cne.fit(
                    X,
                    Y,
                    isSurrogate=isSurrogate,
                    client_graph_info=client_graph_info,
                    client_attraction_thred=client_attraction_thred,
                    isCent=isCent,
                    add_both=add_both,
                    gpu_id=self.gpu_id,
        )

        return self





    # ------------------------------------------------------------
    #                       Test global model
    # ------------------------------------------------------------
    def test(self, X, graph=None, batch_size=-1, seed=-1, loss_aggregation='mean'):

        # Load embedding engine
        self.cne = ContrastiveEmbedding(
                                self.model,
                                n_epochs=self.n_epochs,
                                seed=seed,
                                loss_aggregation=self.loss_aggregation,
                                anneal_lr=self.anneal_lr,
                                batch_size=batch_size,
                                **self.kwargs)

        # pass the similarity graph with annoy if none is given
        self.neighbor_mat = graph.tocsr()

        # create data loader
        self.dataloader = FastTensorDataLoader(self.neighbor_mat,
                                               shuffle=False,
                                               batch_size=batch_size,
                                               on_gpu=self.on_gpu,
                                               seed=seed,
                                               gpu_id=self.gpu_id,)

        _, loss, loss_pos, loss_neg = self.cne.compute_test_loss(self.dataloader, loss_aggregation=loss_aggregation)
        return loss, loss_pos, loss_neg


