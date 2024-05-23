import numpy as np
import random
import os
import pickle
import argparse
import sys
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split





def load_MACOSKO():
    """
    #dataset_address = 'http://file.biolab.si/opentsne/macosko_2015.pkl.gz'
    # https://opentsne.readthedocs.io/en/latest/examples/01_simple_usage/01_simple_usage.html
    # also see https://github.com/berenslab/rna-seq-tsne/blob/master/umi-datasets.ipynb

    Returns
    -------
    [type]
        [description]
    """
    with open("data/macosko_2015.pkl", "rb") as f:
        data = pickle.load(f)

    x = data["pca_50"].astype(np.float32)
    y = data["CellType1"].astype(str)

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    np.shape(X_train)

    n_valid = 10000
    X_valid = X_train[-n_valid:]
    Y_valid = Y_train[-n_valid:]
    X_train = X_train[:-n_valid]
    Y_train = Y_train[:-n_valid]

    enc = OrdinalEncoder()
    Y_train = enc.fit_transform([[i] for i in Y_train]).flatten()
    Y_valid = enc.fit_transform([[i] for i in Y_valid]).flatten()
    Y_test = enc.fit_transform([[i] for i in Y_test]).flatten()

    print(np.shape(X_train), np.shape(X_test), np.shape(Y_train), np.shape(Y_test))
    return X_train, X_test, Y_train, Y_test





def get_rnaseq_dataset(args, dataset_name, folder_path, isCent=False):
    X_train, Y_train = None, None
    client_data, client_labels, dict_users = None, None, None

    if dataset_name == 'rnaseq':
        X_train, _, Y_train, _ = load_MACOSKO()

        if not isCent:
            if args.iid:
                print('IID not implemented ...')
                # client_data, client_labels, dict_users = mnist_iid(num_users=args.n_users, seed=args.seed,
                #                                                    path=folder_path)
            else:
                client_data, client_labels, dict_users = rnaseq_noniid(num_users=args.n_users, method="dir",
                                                                      num_data=len(X_train), alpha=args.alpha, seed=args.seed,
                                                                      path=folder_path)

    else:
        print(f'Dataset {dataset_name} Not implemented yet...')
        sys.exit(0)

    if isCent:
        return X_train, Y_train
    return X_train, Y_train, client_data, client_labels, dict_users






def rnaseq_noniid(num_users, method="dir", num_data=60000, alpha=0.3, seed=42, path=''):
    np.random.seed(seed)
    random.seed(seed)

    train_images, _, train_labels, _ = load_MACOSKO()
    # train_images = train_images.reshape(train_images.shape[0], -1)

    n_classes = len(np.unique(train_labels))

    dataset = train_images    
    labels = train_labels

    _lst_sample = 0  # if num_users > 10 else 10

    min_size = 0
    K = n_classes
    y_train = labels

    _lst_sample = 0

    least_idx = np.zeros((num_users, n_classes, _lst_sample), dtype=np.int64)
    for i in range(n_classes):
        idx_i = np.random.choice(np.where(labels == i)[0], num_users * _lst_sample, replace=False)
        least_idx[:, i, :] = idx_i.reshape((num_users, _lst_sample))
    least_idx = np.reshape(least_idx, (num_users, -1))

    least_idx_set = set(np.reshape(least_idx, (-1)))
    # least_idx_set = set([])
    server_idx = np.random.choice(list(set(range(num_data)) - least_idx_set), num_data - num_data, replace=False)
    local_idx = np.array([i for i in range(num_data) if i not in server_idx and i not in least_idx_set])

    N = y_train.shape[0]
    net_dataidx_map = {}
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    while min_size < n_classes:
        idx_batch = [[] for _ in range(num_users)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            idx_k = [id for id in idx_k if id in local_idx]

            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = idx_batch[j]
        dict_users[j] = np.concatenate((dict_users[j], least_idx[j]), axis=0)

    client_data = []
    client_labels = []
    cnts_dict = {}
    # with open("data_%d_u%d_%s.txt"%(num_data, num_users, method), 'w') as f:
    with open(os.path.join(path, f"RNASEQ_data{num_data}_u{num_users}_{method}_alpha{alpha}_seed{seed}.txt"), 'w') as f:
        for i in range(num_users):
            labels_i = labels[dict_users[i]]
            cnts = np.array([np.count_nonzero(labels_i == j) for j in range(n_classes)])
            cnts_dict[i] = cnts
            f.write("User %s: %s sum: %d\n" % (i, " ".join([str(cnt) for cnt in cnts]), sum(cnts)))

    for indices in dict_users.values():
        indices = list(indices)
        client_data.append( np.take(dataset, indices, axis=0) )
    for i in range(num_users):
        client_labels.append( labels[dict_users[i]] )
    return client_data, client_labels, dict_users






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.iid = False
    args.n_users = 20
    args.seed = 42
    args.alpha = 0.5

    dataset_name = 'rnaseq'

    X_train, Y_train, client_data, client_labels, dict_users = get_rnaseq_dataset(args, dataset_name,
                                                                                 folder_path='', isCent=False)




