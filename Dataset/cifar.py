import numpy as np
import random
import torch
import os
import torchvision
import pickle




def load_cifar10():

    # trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=None) 
    # x_train, y_train = trainset.data.astype(np.float32), torch.Tensor(np.array(trainset.targets))

    # testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=None)
    # x_test, y_test = testset.data.astype(np.float32), torch.Tensor(np.array(testset.targets))

    with open('Dataset/cifar_resnet34/cifar10_train.pkl', 'rb') as file:
        [x_train, y_train] = pickle.load(file)

    with open('Dataset/cifar_resnet34/cifar10_test.pkl', 'rb') as file:
        [x_test, y_test] = pickle.load(file)

    x_train = x_train.numpy().astype(np.float32) #/ 255.
    x_test = x_test.numpy().astype(np.float32) #/ 255.

    return x_train, y_train




def load_test_cifar10():
    with open('Dataset/cifar_resnet34/cifar10_test.pkl', 'rb') as file:
        [x_test, y_test] = pickle.load(file)

    x_test = x_test.numpy().astype(np.float32) #/ 255.

    return x_test, y_test




def load_test_cifar100():
    with open('Dataset/cifar_resnet34/cifar100_test.pkl', 'rb') as file:
        [x_test, y_test] = pickle.load(file)

    x_test = x_test.numpy().astype(np.float32) #/ 255.

    return x_test, y_test 





def load_cifar100():

    # trainset = torchvision.datasets.CIFAR100(root='data', train=True, download=True, transform=None)
    # x_train, y_train = trainset.data.astype(np.float32), torch.Tensor(np.array(trainset.targets))

    # testset = torchvision.datasets.CIFAR100(root='data', train=False, download=True, transform=None)
    # x_test, y_test = testset.data.astype(np.float32), torch.Tensor(np.array(testset.targets))

    with open('Dataset/cifar_resnet34/cifar100_train.pkl', 'rb') as file:
        [x_train, y_train] = pickle.load(file)

    with open('Dataset/cifar_resnet34/cifar100_test.pkl', 'rb') as file:
        [x_test, y_test] = pickle.load(file)

    x_train = x_train.numpy().astype(np.float32) #/ 255.
    x_test = x_test.numpy().astype(np.float32) #/ 255.

    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)

    # x_train = x_train / np.max(x_train)
    # x_test = x_test / np.max(x_test)
    return x_train, y_train






def cifar10_iid(num_users, seed=42, path=''):
    np.random.seed(seed)
    random.seed(seed)

    train_images, train_labels = load_cifar10()
    train_images = train_images.reshape(train_images.shape[0], -1)
    dataset = np.array(train_images)

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    client_data = []
    client_labels = []
    cnts_dict = {}
    with open(os.path.join(path, "CIFAR10_IID_u%d_seed%d.txt"%(num_users, seed)), 'w') as f:
        for i in range(num_users):
            labels_i = np.array(train_labels)[list(dict_users[i])]
            cnts = np.array( [np.count_nonzero(labels_i == j) for j in range(10)] )
            cnts_dict[i] = cnts
            f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))

    for indices in dict_users.values():
        indices = list(indices)
        client_data.append( np.take(dataset, indices, axis=0) )
    for i in range(num_users):
        client_labels.append( np.array(train_labels)[list(dict_users[i])] )
    return client_data, client_labels, dict_users




def cifar100_iid(num_users, seed=42, path=''):
    np.random.seed(seed)
    random.seed(seed)

    train_images, train_labels = load_cifar100()
    train_images = train_images.reshape(train_images.shape[0], -1)
    dataset = train_images

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    client_data = []
    client_labels = []
    cnts_dict = {}
    with open(os.path.join(path, "CIFAR100_IID_u%d_seed%d.txt"%(num_users, seed)), 'w') as f:
        for i in range(num_users):
            labels_i = np.array(train_labels)[list(dict_users[i])]
            cnts = np.array( [np.count_nonzero(labels_i == j) for j in range(10)] )
            cnts_dict[i] = cnts
            f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))

    for indices in dict_users.values():
        indices = list(indices)
        client_data.append( np.take(dataset, indices, axis=0) )
    for i in range(num_users):
        client_labels.append( np.array(train_labels)[list(dict_users[i])] )
    return client_data, client_labels, dict_users




def cifar10_noniid(num_users, method="dir", num_data=50000, alpha=0.3, seed=42, path=''):
    np.random.seed(seed)
    random.seed(seed)

    train_images, train_labels = load_cifar10()
    train_images = train_images.reshape(train_images.shape[0], -1)

    dataset = train_images   #(50000,784)
    labels = train_labels

    _lst_sample = 0  # if num_users > 10 else 10

    min_size = 0
    K = 10
    y_train = labels

    _lst_sample = 0

    least_idx = np.zeros((num_users, 10, _lst_sample), dtype=np.int64)
    for i in range(10):
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

    while min_size < 10:
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
    with open(os.path.join(path, f"CIFAR10_data{num_data}_u{num_users}_{method}_alpha{alpha}_seed{seed}.txt"), 'w') as f:
        for i in range(num_users):
            labels_i = labels[dict_users[i]]
            cnts = np.array([np.count_nonzero(labels_i == j) for j in range(10)])
            cnts_dict[i] = cnts
            f.write("User %s: %s sum: %d\n" % (i, " ".join([str(cnt) for cnt in cnts]), sum(cnts)))

    for indices in dict_users.values():
        indices = list(indices)
        client_data.append( np.take(dataset, indices, axis=0) )
    for i in range(num_users):
        client_labels.append( labels[dict_users[i]] )
    return client_data, client_labels, dict_users




def cifar100_noniid(num_users, method="dir", num_data=50000, alpha=0.3, seed=42, path=''):
    np.random.seed(seed)
    random.seed(seed)

    train_images, train_labels = load_cifar100()
    train_images = train_images.reshape(train_images.shape[0], -1)

    dataset = train_images   #(50000,784)
    labels = train_labels

    _lst_sample = 0  # if num_users > 10 else 10

    min_size = 0
    K = 100
    y_train = labels

    _lst_sample = 0

    least_idx = np.zeros((num_users, 100, _lst_sample), dtype=np.int64)
    for i in range(10):
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

    while min_size < 10:
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
    with open(os.path.join(path, f"CIFAR100_data{num_data}_u{num_users}_{method}_alpha{alpha}_seed{seed}.txt"), 'w') as f:
        for i in range(num_users):
            labels_i = labels[dict_users[i]]
            cnts = np.array([np.count_nonzero(labels_i == j) for j in range(100)])
            cnts_dict[i] = cnts
            f.write("User %s: %s sum: %d\n" % (i, " ".join([str(cnt) for cnt in cnts]), sum(cnts)))

    for indices in dict_users.values():
        indices = list(indices)
        client_data.append( np.take(dataset, indices, axis=0) )
    for i in range(num_users):
        client_labels.append( labels[dict_users[i]] )
    
    return client_data, client_labels, dict_users

