import numpy as np
import random
import copy
import os
import torchvision


def range_split(a, n):
    # a: range; n: count
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def get_xy_splits(xx, yy, n):
    xx = sorted(xx)
    yy = sorted(yy)
    splits = []
    k1, m1 = divmod(len(xx), n)
    k2, m2 = divmod(len(yy), n)
    for i in range(1,n):
        splits.append([xx[(i)*k1], yy[(i)*k2]])
    return splits




def load_zebrafish(root_path):
    root_path = os.path.join(root_path, "zebrafish")
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    try:
        x = np.load(os.path.join(root_path, "zfish.data.npy"))
        y = np.load(os.path.join(root_path, "zfish.labels.npy"))
    except FileNotFoundError:
        # download
        print("Downloading zebrafish data...")
        url = "https://kleintools.hms.harvard.edu/paper_websites/wagner_zebrafish_timecourse2018/WagnerScience2018.h5ad"
        file_name = "WagnerScience2018.h5ad"
        file_path = os.path.join(root_path, file_name)
        urllib.request.urlretrieve(url, file_path)

        print("Preprocessing zebrafish data...")
        # preprocess
        X, stage, alt_c = zfish_preprocess(file_path)
        np.save(os.path.join(root_path, "zfish.data.npy"), X)
        np.save(os.path.join(root_path, "zfish.labels.npy"), stage)
        np.save(os.path.join(root_path, "zfish.altlabels.npy"), alt_c)
        print("...done.")

        x = X
        y = stage
    return x, y




def normalize_image(image):
    # Convert the image to float in case it isn't
    image = image.astype(np.float32)

    # Find the minimum and maximum values in the image
    min_val = np.min(image)
    max_val = np.max(image)

    # Normalize the image to the 0-1 range
    normalized_image = (image - min_val) / (max_val - min_val)

    return normalized_image




def load_dnaseq():
    x_train, y_train = load_zebrafish('data')
    x_train = normalize_image(x_train)
    return x_train, y_train



def split_rnaseq_dataset(args, folder_path):
    n_clients = args.n_users
    ccc = args.n_classes

    np.random.seed(args.seed)

    # Load MNIST dataset
    train_images, train_labels = load_dnaseq()
    # train_images = train_images.reshape(train_images.shape[0], -1)

    data = train_images  # (63530,50)
    labels = train_labels

    int_list = copy.deepcopy(labels)

    n_data = len(data)
    n_classes = len(np.unique(labels))

    for j, name in enumerate(np.unique(labels)):
        founds = np.where(labels == name)[0]
        int_list[founds] = j

    int_list = np.array(int_list, dtype=int)
    labels = int_list

    # Assign classes to clients:
    client_classes = {}
    class_set = []

    # Initialize a count array to keep track of how many times each class is selected
    class_counts = np.zeros(n_classes)

    for i in range(n_clients):
        available_classes = list(range(n_classes))
        selected_classes = []

        while len(selected_classes) < ccc:
            # Weight classes by their inverse frequency (less frequent classes are more likely to be chosen)
            weights = 1 / (class_counts[available_classes] + 1)
            weights /= weights.sum()

            # Randomly select one class based on the weights
            selected_class = np.random.choice(available_classes, p=weights)
            selected_classes.append(selected_class)

            # Update class counts and remove the selected class from available classes
            class_counts[selected_class] += 1
            available_classes.remove(selected_class)

        client_classes[i] = selected_classes
        class_set += list(selected_classes)


    class_splits = {}
    for c in range(n_classes):
        n_c = len(np.where(labels == c)[0])
        n_sites = class_set.count(c)
        class_splits[c] = split_samples_over_class(n_samples=n_c, n_sites=n_sites, seed=args.seed)

    # Split dataset for each client
    inputs = []
    n_sum = 0
    for cid in client_classes.keys():
        class_list = client_classes[cid]
        num_samples_list = []
        for cls in class_list:
            num_samples_list.append(class_splits[cls][0])
            n_sum += class_splits[cls][0]
            class_splits[cls].pop(0)
        inputs.append({"class": class_list, "num_samples": num_samples_list})

    clients_data, clients_labels = [], []
    splits = split_rnaseq_by_input(inputs)

    for i, (img, lab) in enumerate(splits):
        clients_data.append(img)
        clients_labels.append(lab)

    # Write clients data summary to a text file
    output_file = f"RNAseq_data{n_data}_u{n_clients}_c{ccc}_seed{args.seed}.txt"
    n_sum = 0
    with open(os.path.join(folder_path, output_file), "w") as file:
        for i in range(n_clients):
            client_data, client_labels = clients_data[i], clients_labels[i]

            # Count occurrences of each digit label
            label_counts = [np.sum(client_labels == j) for j in range(10)]
            sum_labels = np.sum(label_counts)

            # Formatting and writing to file
            label_counts_str = ' '.join(str(count) for count in label_counts)
            line = f"User {i}: {label_counts_str} sum: {sum_labels}\n"
            file.write(line)
            n_sum += sum_labels

    assert n_sum == n_data

    return data, labels, clients_data, clients_labels







def load_mnist():
    # load MNIST
    mnist_train = torchvision.datasets.MNIST('data', train=True, download=True, transform=None)
    x_train, y_train = mnist_train.data.float().numpy(), mnist_train.targets

    mnist_test = torchvision.datasets.MNIST('data', train=False, download=True, transform=None)
    x_test, y_test = mnist_test.data.float().numpy(), mnist_test.targets

    x_train = x_train / 255.
    x_test = x_test / 255.
    return x_train, y_train



def load_fashmnist():
    # load F-MNIST
    fmnist_train = torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=None)
    x_train, y_train = fmnist_train.data.float().numpy(), fmnist_train.targets

    fmnist_test = torchvision.datasets.FashionMNIST('data', train=False, download=True, transform=None)
    x_test, y_test = fmnist_test.data.float().numpy(), fmnist_test.targets

    x_train = x_train / 255.
    x_test = x_test / 255.
    return x_train, y_train



def subset_byClass(digit_labels, num_data=500, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    #
    train_images, train_labels = load_mnist()
    train_images = train_images.reshape((train_images.shape[0], -1))

    inds_list = []
    inds_all = []
    for c in digit_labels:
        inds = random.sample(list(np.where(train_labels == c)[0]), num_data)
        inds_list.append(inds)
        inds_all += inds

    subdataset = train_images[inds_all]
    sublabels = train_labels[inds_all]

    return subdataset, sublabels #, subdataset7, sublabels7, subdataset9, sublabels9



def subset_byClass_penDigits(train_images, train_labels, digit_labels, num_data=500, seed=42):
    np.random.seed(seed)
    import random
    random.seed(seed)
    #
    # train_images, train_labels = load_mnist()
    # train_images = train_images.reshape((train_images.shape[0], -1))

    inds_list = []
    inds_all = []
    for c in digit_labels:
        inds = random.sample(list(np.where(train_labels == c)[0]), num_data)
        inds_list.append(inds)
        inds_all += inds

    subdataset = train_images[inds_all]
    sublabels = train_labels[inds_all]

    return subdataset, sublabels



def mnist_iid(num_users, seed=42, path=''):
    np.random.seed(seed)
    import random
    random.seed(seed)

    train_images, train_labels = load_mnist()
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
    with open(os.path.join(path, "MNIST_IID_u%d_seed%d.txt"%(num_users, seed)), 'w') as f:
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




def fashmnist_iid(num_users, seed=42, path=''):
    np.random.seed(seed)
    random.seed(seed)

    train_images, train_labels = load_fashmnist()
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
    with open(os.path.join(path, "FashMNIST_IID_u%d_seed%d.txt"%(num_users, seed)), 'w') as f:
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




def mnist_noniid(num_users, method="dir", num_data=60000, alpha=0.3, seed=42, path=''):
    np.random.seed(seed)
    random.seed(seed)

    train_images, train_labels = load_mnist()
    train_images = train_images.reshape(train_images.shape[0], -1)

    dataset = train_images   #(60000,784)
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
    with open(os.path.join(path, f"MNIST_data{num_data}_u{num_users}_{method}_alpha{alpha}_seed{seed}.txt"), 'w') as f:
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



def fashmnist_noniid(num_users, method="dir", num_data=60000, alpha=0.3, seed=42, path=''):
    np.random.seed(seed)
    random.seed(seed)

    train_images, train_labels = load_fashmnist()
    train_images = train_images.reshape(train_images.shape[0], -1)

    dataset = train_images   #(60000,784)
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
    with open(os.path.join(path, f"FashMNIST_data{num_data}_u{num_users}_{method}_alpha{alpha}_seed{seed}.txt"), 'w') as f:
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




def mnist_subset_iid(dataset, labels, num_users, seed=42, path=''):
    np.random.seed(seed)
    import random
    random.seed(seed)

    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])

    client_data = []
    client_labels = []
    cnts_dict = {}
    with open(os.path.join(path, "tinyMNIST_IID_u%d_seed%d.txt"%(num_users, seed)), 'w') as f:
        for i in range(num_users):
            labels_i = np.array(labels)[list(dict_users[i])]
            cnts = np.array( [np.count_nonzero(labels_i == j) for j in range(10)] )
            cnts_dict[i] = cnts
            f.write("User %s: %s sum: %d\n"%(i, " ".join([str(cnt) for cnt in cnts]), sum(cnts) ))

    for indices in dict_users.values():
        indices = list(indices)
        client_data.append( np.take(dataset, indices, axis=0) )
    for i in range(num_users):
        client_labels.append( np.array(labels)[list(dict_users[i])] )
    return client_data, client_labels, dict_users



def mnist_subset_noniid(dataset, labels, num_users, method="dir", num_data=60000, alpha=0.3, seed=42, path=''):
    np.random.seed(seed)
    import random
    random.seed(seed)

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
    with open(os.path.join(path, f"tinyMNIST_data{num_data}_u{num_users}_{method}_alpha{alpha}_seed{seed}.txt"), 'w') as f:
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



def client_data_by_class(X_train, Y_train, num_per_class=100, labels=None):
    client_data, client_labels = [], []
    dict_users = {}

    # X_train, Y_train = load_mnist()
    X_train = X_train.reshape(X_train.shape[0], -1)

    classes = np.unique(Y_train)

    if labels is None:
        for label in classes:
            inds = np.where(Y_train == label)[0]
            inds = inds[:num_per_class]
            data = X_train[inds]
            client_data.append(data)
            client_labels.append(Y_train[inds])
            dict_users[label] = inds
    else:
        for label in labels:
            inds = np.where(Y_train == label)[0]
            inds = inds[:num_per_class]
            data = X_train[inds]
            client_data.append(data)
            client_labels.append(Y_train[inds])
            dict_users[label] = inds

    return client_data, client_labels, dict_users



def split_dataset(images, labels, cnt_list):
    client_images, client_labels = [], []
    start, end = 0, 0

    for i, cnt in enumerate(cnt_list):
        end = start + cnt

        images_curr = images[start:end,:]
        client_images.append(images_curr)

        labels_curr = labels[start:end]
        client_labels.append(labels_curr)

        start += cnt
    return client_images, client_labels






if __name__ == '__main__':

    # client_dataset1 = sample_3d_bison_iid(5)
    # client_dataset2 = sample_mnist_iid(4)

    # client_data = sample_mnist_iid(num_users=5)

    sample_mnist_iid(num_users=5)
    sample_mnist_noniid(num_users=5, method="dir", num_data=60000, alpha=0.3, seed=42)

    print()

