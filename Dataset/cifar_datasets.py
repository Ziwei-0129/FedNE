import sys
import argparse
import numpy as np
import urllib
from Dataset.mnist import load_mnist, subset_byClass, client_data_by_class
from Dataset.mnist import mnist_iid, mnist_noniid, mnist_subset_iid, mnist_subset_noniid, split_dataset
from Dataset.cifar import load_cifar10, load_cifar100, cifar10_iid, cifar10_noniid, cifar100_iid, cifar100_noniid





def get_cifar10_dataset(args, dataset_name, folder_path, isCent=False):

    client_data, client_labels, dict_users = None, None, None

    X_train, Y_train = load_cifar10()
    X_train = X_train.reshape(X_train.shape[0], -1)

    if not isCent:
        if args.iid:
            client_data, client_labels, dict_users = cifar10_iid(num_users=args.n_users, seed=args.seed,
                                                               path=folder_path)
        else:
            client_data, client_labels, dict_users = cifar10_noniid(num_users=args.n_users, method="dir",
                                                    num_data=50000, alpha=args.alpha, seed=args.seed, path=folder_path)

    if isCent:
        return X_train, Y_train

    return X_train, Y_train, client_data, client_labels, dict_users






def get_cifar100_dataset(args, dataset_name, folder_path, isCent=False):

    client_data, client_labels, dict_users = None, None, None

    X_train, Y_train = load_cifar100()
    X_train = X_train.reshape(X_train.shape[0], -1)

    if not isCent:
        if args.iid:
            client_data, client_labels, dict_users = cifar100_iid(num_users=args.n_users, seed=args.seed,
                                                               path=folder_path)
        else:
            client_data, client_labels, dict_users = cifar100_noniid(num_users=args.n_users, method="dir",
                                                    num_data=50000, alpha=args.alpha, seed=args.seed, path=folder_path)

    if isCent:
        return X_train, Y_train

    return X_train, Y_train, client_data, client_labels, dict_users






def split_cifar10_by_input(inputs):

    x_train, y_train = load_cifar10()
    x_train = x_train.reshape(x_train.shape[0], -1)

    # Create a mask of zeros for the entire training set
    mask = np.zeros_like(y_train, dtype=bool)

    results = []
    for inp in inputs:
        current_split_x = []
        current_split_y = []
        for c, num_samples in zip(inp["class"], inp["num_samples"]):
            # Create a mask for the current class and exclude already chosen samples
            current_class_mask = (y_train == c) & ~mask
            current_class_indices = np.where(current_class_mask)[0]

            if num_samples > len(current_class_indices):
                raise ValueError(
                    f"Requested {num_samples} samples for class {c}, but only {len(current_class_indices)} available.")

            chosen_indices = np.random.choice(current_class_indices, size=num_samples, replace=False)
            current_split_x.append(x_train[chosen_indices])
            current_split_y.append(y_train[chosen_indices])

            # Update the main mask to mark the chosen samples
            mask[chosen_indices] = True

        # Concatenate the selected samples for this split and append to results
        results.append((np.concatenate(current_split_x, axis=0), np.concatenate(current_split_y, axis=0)))

    return results





def split_cifar100_by_input(inputs):

    x_train, y_train = load_cifar100()
    x_train = x_train.reshape(x_train.shape[0], -1)

    # Create a mask of zeros for the entire training set
    mask = np.zeros_like(y_train, dtype=bool)

    results = []
    for inp in inputs:
        current_split_x = []
        current_split_y = []
        for c, num_samples in zip(inp["class"], inp["num_samples"]):
            # Create a mask for the current class and exclude already chosen samples
            current_class_mask = (y_train == c) & ~mask
            current_class_indices = np.where(current_class_mask)[0]

            if num_samples > len(current_class_indices):
                raise ValueError(
                    f"Requested {num_samples} samples for class {c}, but only {len(current_class_indices)} available.")

            chosen_indices = np.random.choice(current_class_indices, size=num_samples, replace=False)
            current_split_x.append(x_train[chosen_indices])
            current_split_y.append(y_train[chosen_indices])

            # Update the main mask to mark the chosen samples
            mask[chosen_indices] = True

        # Concatenate the selected samples for this split and append to results
        results.append((np.concatenate(current_split_x, axis=0), np.concatenate(current_split_y, axis=0)))

    return results


