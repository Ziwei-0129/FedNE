import random
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torch.utils.data import Sampler

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




def generate_batch(n_local, n_batches, batch_size):
    indices = np.arange(n_local, dtype=np.int64)
    np.random.seed(int(time.time()))
    rand_inds = np.random.choice(indices, n_local, replace=False)

    if batch_size == -1 and n_batches == 1:  # full batch
        return [rand_inds]

    batches_arr = []
    for i in range(0, len(rand_inds), batch_size):
        batches_arr.append(rand_inds[i:i + batch_size])
        if 0 < n_batches == len(batches_arr):  # Break if n_batches batches are created
            break
    return batches_arr



class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]



def generate_batch222(targets, n_local, n_batches, batch_size):
    indices = torch.arange(n_local)
    torch.manual_seed(int(time.time()))

    dataset = MyDataset(indices, targets)

    sampler = StratifiedSampler(dataset)

    # # Automatically generate class_to_index mapping
    # unique_targets = torch.unique(targets, sorted=True)
    # class_to_index = {target.item(): index for index, target in enumerate(unique_targets)}
    #
    # # Calculate weights for each class
    # class_sample_count = torch.tensor([(targets == t).sum() for t in unique_targets])
    # weight = 1. / class_sample_count.float()
    #
    # # Map class labels in targets to weight indices
    # samples_weight = torch.tensor([weight[class_to_index[t.item()]] for t in targets])
    #
    # # Create WeightedRandomSampler
    # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Create DataLoader with the sampler
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return dataloader





class StratifiedSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))

        # Count the occurrences of each class
        class_counts = {}
        for _, label in self.dataset:
            label = label.item()
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        # Calculate weights for each class
        weights = {label: 1.0 / count for label, count in class_counts.items()}

        # Create a label to index mapping
        label_to_index = {label: i for i, label in enumerate(sorted(class_counts))}

        # Assign a weight to each sample
        # self.sample_weights = [weights[label] for _, label in self.dataset]
        self.sample_weights = [weights[sorted(class_counts)[label_to_index[label.item()]]] for _, label in self.dataset]

    def __iter__(self):
        # Shuffle indices at the start of each epoch
        weighted_indices = [(index, weight) for index, weight in zip(self.indices, self.sample_weights)]
        weighted_indices.sort(key=lambda x: torch.rand(1))  # Shuffle with a random key

        # Yield indices based on the shuffled order
        for index, _ in weighted_indices:
            yield index

    def __len__(self):
        return len(self.dataset)





def generate_batch_2(n_local, n_batches, batch_size):
    indices = np.array([*range(n_local)], dtype=np.int64)

    if batch_size == -1 and n_batches == 1:    # full batch
        np.random.seed(int(time.time()))
        rand_inds = np.random.choice(indices, n_local, replace=False)
        batches_arr = [rand_inds]

    elif n_batches == -1:   # run full epoch
        batches_arr = []

        # split into batches
        np.random.seed(int(time.time()))
        rand_inds = np.random.choice(indices, n_local, replace=False)
        batches = [rand_inds[i:i + batch_size] for i in range(0, len(rand_inds), batch_size)]

        num_batches = len(batches)

        for i in range(num_batches):
            batches_arr.append(batches[i])

    else:
        batches_arr = []

        # split into batches
        np.random.seed(int(time.time()))
        rand_inds = np.random.choice(indices, n_local, replace=False)
        batches = [rand_inds[i:i + batch_size] for i in range(0, len(rand_inds), batch_size)]

        num_batches = len(batches)
        if num_batches < n_batches:
            n_batches = num_batches

        for i in range(n_batches):
            batches_arr.append(batches[i])

    return batches_arr





def make_neghbors_upperbound(item_inds, neg_inds_candidates):
    negative_samples = 5
    b = len(item_inds)
    neg_inds_candidates = torch.tensor(neg_inds_candidates, device=device)

    rand_num = random.randint(0, 10000)
    torch.manual_seed(rand_num)
    rand_ints = torch.randint(0, len(neg_inds_candidates), (b * negative_samples,), device=device)
    neg_inds = neg_inds_candidates[rand_ints]
    return neg_inds




def make_neg_features(model, neg_inds_cands, b):
    negative_samples = 5
    neg_inds_cands = torch.tensor(neg_inds_cands, device=device)
    rand_num = random.randint(0, 10000)
    torch.manual_seed(rand_num)
    rand_ints = torch.randint(0, len(neg_inds_cands), (b * negative_samples,), device=device)
    neg_inds = neg_inds_cands[rand_ints]
    fea_nonnbrs = model(neg_inds)
    fea_nonnbrs = torch.reshape(fea_nonnbrs, (b, negative_samples, 2))
    return fea_nonnbrs




def make_neg_inds(neg_inds_cands, n_rows):
    negative_samples = m = 5
    neg_inds_cands = torch.tensor(neg_inds_cands, device=device)
    torch.manual_seed(int(time.time()))
    rand_ints = torch.randint(0, len(neg_inds_cands), (n_rows * negative_samples,), device=device)
    neg_inds = neg_inds_cands[rand_ints]
    neg_inds = torch.reshape(neg_inds, (n_rows, negative_samples))
    return neg_inds




def criterion_client_pos(model, batch_inds, pos_inds_candidates, loss_mode='mean'):

    noise_in_estimator = torch.tensor(1.0, device=device)
    eps = torch.tensor(1.0, device=device)
    clamp_low, clamp_high = 0.0001, 1.0

    loss_pos = torch.zeros((len(pos_inds_candidates), 6), device=device)


    for ind, nbrs_list in enumerate(pos_inds_candidates):

        data_ind = batch_inds[ind]
        n_neigs = len(nbrs_list)
        # n_neigs = 1

        random.seed(time.time())
        # nbrs_list = [random.choice(nbrs_list)]


        if n_neigs != 0:
            inds_orig = [data_ind] * n_neigs

            neigh_inds = torch.zeros((n_neigs, 1), dtype=torch.int32, device=device)
            neigh_inds[:, 0] = torch.tensor(np.array(nbrs_list), dtype=torch.int32, device=device)
            neigh_inds_flatten = torch.flatten(neigh_inds)

            neighbors = model(neigh_inds_flatten)
            neighbors = torch.reshape(neighbors, (n_neigs, 1, 2))

            inds_orig = torch.tensor(np.array(inds_orig), device=device)
            origs = model(inds_orig)

            dists = ((origs[:, None] - neighbors) ** 2).sum(axis=2)
            estimator = 1 / (1 + noise_in_estimator * (dists + eps))

            loss_pos_all = - torch.log(estimator.clamp(clamp_low, clamp_high))
                
            loss_pos[ind, 0, None] = torch.mean(loss_pos_all)
            # loss_pos[ind, 0, None] = torch.sum(loss_pos_all) #/ scale_mat_pos[ind] #!!!!!!!!!!!!!!!!!!!!!

        else:
            loss_pos[ind, 0, None] = torch.tensor(0.0, device=device)


    if loss_mode == 'sum':
        loss_pos = loss_pos.sum()
    elif loss_mode is None:
        return loss_pos.mean(1)
    else:
        loss_pos = loss_pos.mean()
    return loss_pos





def criterion_client_neg(model, batch_inds, pos_inds_candidates, neg_inds_candidates, loss_mode='mean'):

    noise_in_estimator = torch.tensor(1.0, device=device)
    eps = torch.tensor(1.0, device=device)
    clamp_low, clamp_high = 0.0001, 1.0

    n_rows = 0
    inds_orig_all = []

    for ind, nbrs_list in enumerate(pos_inds_candidates):

        data_ind = batch_inds[ind]

        n_rows += 1
        inds_orig_all += [data_ind]

    loss_neg = torch.zeros((n_rows, 6), device=device)

    inds_neg_all = make_neg_inds(neg_inds_candidates, n_rows)

    neigh_inds = torch.tensor(inds_neg_all, dtype=torch.int32, device=device)
    neigh_inds_flatten = torch.flatten(neigh_inds)

    neighbors = model(neigh_inds_flatten)
    neighbors = torch.reshape(neighbors, (n_rows, 5, 2))

    inds_orig_all = torch.tensor(inds_orig_all, dtype=torch.int32, device=device)
    origs = model(inds_orig_all)

    dists = ((origs[:, None] - neighbors) ** 2).sum(axis=2)
    estimator = 1 / (1 + noise_in_estimator * (dists + eps))

    loss_neg[:, 1:] = - torch.log((1 - estimator).clamp(clamp_low, clamp_high))

    if loss_mode == 'sum':
        loss_neg = loss_neg.sum()
    elif loss_mode is None:
        return loss_neg.mean(1)
    else:
        # loss_neg = loss_neg.mean(1)
        loss_neg = loss_neg.mean()

    return loss_neg





def criterion_attraction(model, batch_inds, neigh_inds, loss_mode='mean'):

    noise_in_estimator = torch.tensor(1.0, device=device)
    eps = torch.tensor(1.0, device=device)
    clamp_low, clamp_high = 0.0001, 1.0

    neighbors = model(neigh_inds)
    origins = model(batch_inds)

    dists = ((origins - neighbors) ** 2).sum(axis=1)
    estimator = 1 / (1 + noise_in_estimator * (dists + eps))
    loss_pos_all = - torch.log(estimator.clamp(clamp_low, clamp_high))

    if loss_mode == 'sum':
        loss_pos = loss_pos_all.sum()
    elif loss_mode is None:
        return loss_pos_all
    else:
        loss_pos = loss_pos_all.mean()

    return loss_pos






def criterion_repulsion(model, batch_inds, negative_inds, loss_mode='mean'):

    noise_in_estimator = torch.tensor(1.0, device=device)
    eps = torch.tensor(1.0, device=device)
    clamp_low, clamp_high = 0.0001, 1.0

    negative_inds_flatten = torch.flatten(negative_inds)
    negatives = model(negative_inds_flatten)
    negatives = torch.reshape(negatives, (len(negative_inds), 5, 2))

    origins = model(batch_inds)

    dists = ((origins[:, None] - negatives) ** 2).sum(axis=2)
    estimator = 1 / (1 + noise_in_estimator * (dists + eps))
    loss_neg = - torch.log((1 - estimator).clamp(clamp_low, clamp_high))

    if loss_mode == 'sum':
        loss_neg = loss_neg.sum()
    elif loss_mode is None:
        # return loss_neg.sum(1)
        return loss_neg.mean(1)
    else:
        loss_neg = loss_neg.mean()

    return loss_neg


