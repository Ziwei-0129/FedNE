import sys
import argparse
import numpy as np
import urllib
from Dataset.mnist import load_mnist, subset_byClass, client_data_by_class
from Dataset.mnist import mnist_iid, mnist_noniid, mnist_subset_iid, mnist_subset_noniid, split_dataset
from Dataset.mnist import load_fashmnist, fashmnist_iid, fashmnist_noniid
from Dataset.zfish_preprocess import preprocess as zfish_preprocess
from Dataset.rnaseq_datasets import load_MACOSKO




def split_rnaseq_by_input(inputs):
    x_train, _, y_train, _ = load_MACOSKO()

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






def split_mnist_by_input(inputs):

    x_train, y_train = load_mnist()
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





def split_fashmnist_by_input(inputs):

    x_train, y_train = load_fashmnist()
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




def get_fashmnist_dataset(args, dataset_name, folder_path, isCent=False):
    # X_train, Y_train = None, None
    client_data, client_labels, dict_users = None, None, None
    # if dataset_name == 'mnist':

    X_train, Y_train = load_fashmnist()
    X_train = X_train.reshape(X_train.shape[0], -1)
    if not isCent:
        if args.iid:
            client_data, client_labels, dict_users = fashmnist_iid(num_users=args.n_users, seed=args.seed,
                                                               path=folder_path)
        else:
            client_data, client_labels, dict_users = fashmnist_noniid(num_users=args.n_users, method="dir",
                                                    num_data=60000, alpha=args.alpha, seed=args.seed, path=folder_path)

    if isCent:
        return X_train, Y_train

    return X_train, Y_train, client_data, client_labels, dict_users





def get_mnist_dataset(args, dataset_name, folder_path, isCent=False):
    X_train, Y_train = None, None
    client_data, client_labels, dict_users = None, None, None

    #####################################################################
    #                        Full MNIST dataset
    #####################################################################
    if dataset_name == 'mnist':
        X_train, Y_train = load_mnist()
        X_train = X_train.reshape(X_train.shape[0], -1)

        if not isCent:
            if args.iid:
                client_data, client_labels, dict_users = mnist_iid(num_users=args.n_users, seed=args.seed,
                                                                   path=folder_path)
            else:
                client_data, client_labels, dict_users = mnist_noniid(num_users=args.n_users, method="dir",
                                                                      num_data=60000, alpha=args.alpha, seed=args.seed,
                                                                      path=folder_path)


    elif dataset_name == 'mnist_8client4class_iid':
        inputs = [
            {"class": [0, 1, 2, 3], "num_samples": [300, 300, 300, 300]},
            {"class": [0, 1, 2, 3], "num_samples": [300, 300, 300, 300]},
            {"class": [0, 1, 2, 3], "num_samples": [300, 300, 300, 300]},
            {"class": [0, 1, 2, 3], "num_samples": [300, 300, 300, 300]},
            {"class": [0, 1, 2, 3], "num_samples": [300, 300, 300, 300]},
            {"class": [0, 1, 2, 3], "num_samples": [300, 300, 300, 300]},
            {"class": [0, 1, 2, 3], "num_samples": [300, 300, 300, 300]},
            {"class": [0, 1, 2, 3], "num_samples": [300, 300, 300, 300]},
        ]

        client_data, client_labels = [], []
        splits = split_mnist_by_input(inputs)
        for i, (img, lab) in enumerate(splits):
            client_data.append(img)
            client_labels.append(lab)

        X_train, Y_train = np.vstack(client_data), np.hstack(client_labels)


    elif dataset_name == 'mnist_8client4class_mildnoniid':
        inputs = [
            {"class": [0, 1, 2, 3], "num_samples": [300, 300, 300, 300]},  #0
            {"class": [0, 1, 2, 3], "num_samples": [300, 300, 300, 300]},  #1
            {"class": [0], "num_samples": [1200]},                         #2
            {"class": [1], "num_samples": [1200]},                         #3
            {"class": [2], "num_samples": [1200]},                         #4
            {"class": [3], "num_samples": [1200]},                         #5
            {"class": [0, 1], "num_samples": [600, 600]},                  #6
            {"class": [2, 3], "num_samples": [600, 600]},                  #7
        ]

        client_data, client_labels = [], []
        splits = split_mnist_by_input(inputs)
        for i, (img, lab) in enumerate(splits):
            client_data.append(img)
            client_labels.append(lab)

        X_train, Y_train = np.vstack(client_data), np.hstack(client_labels)



    elif dataset_name == 'mnist_8client4class_noniid':
        inputs = [
            {"class": [0, 1, 2, 3], "num_samples": [600, 600, 0, 0]},
            {"class": [0, 1, 2, 3], "num_samples": [600, 600, 0, 0]},
            {"class": [0, 1, 2, 3], "num_samples": [600, 600, 0, 0]},
            {"class": [0, 1, 2, 3], "num_samples": [600, 600, 0, 0]},
            {"class": [0, 1, 2, 3], "num_samples": [0, 0, 600, 600]},
            {"class": [0, 1, 2, 3], "num_samples": [0, 0, 600, 600]},
            {"class": [0, 1, 2, 3], "num_samples": [0, 0, 600, 600]},
            {"class": [0, 1, 2, 3], "num_samples": [0, 0, 600, 600]},
        ]

        client_data, client_labels = [], []
        splits = split_mnist_by_input(inputs)
        for i, (img, lab) in enumerate(splits):
            client_data.append(img)
            client_labels.append(lab)

        X_train, Y_train = np.vstack(client_data), np.hstack(client_labels)



    elif dataset_name == 'mnist_6client3class_noniid':
        inputs = [
            {"class": [7, 8, 9], "num_samples": [300, 200, 100]},
            {"class": [7, 8, 9], "num_samples": [200, 100, 300]},
            {"class": [7, 8, 9], "num_samples": [100, 300, 200]},
            {"class": [7, 8, 9], "num_samples": [150, 50, 100]},
            {"class": [7, 8, 9], "num_samples": [50, 100, 150]},
            {"class": [7, 8, 9], "num_samples": [100, 150, 50]},
        ]

        client_data, client_labels = [], []
        splits = split_mnist_by_input(inputs)
        for i, (img, lab) in enumerate(splits):
            client_data.append(img)
            client_labels.append(lab)

        X_train, Y_train = np.vstack(client_data), np.hstack(client_labels)



    elif dataset_name == 'mnist_4client4class_noniid':
        inputs = [
            {"class": [0, 8], "num_samples": [300, 200]},
            {"class": [0, 8], "num_samples": [300, 400]},
            {"class": [1, 9], "num_samples": [400, 300]},
            {"class": [1, 9], "num_samples": [200, 300]},
        ]

        client_data, client_labels = [], []
        splits = split_mnist_by_input(inputs)
        for i, (img, lab) in enumerate(splits):
            client_data.append(img)
            client_labels.append(lab)

        X_train, Y_train = np.vstack(client_data), np.hstack(client_labels)




    elif dataset_name == 'mnist_2client1class_iid':
        inputs = [
            {"class": [0], "num_samples": [300]},
            {"class": [0], "num_samples": [200]},
        ]

        client_data, client_labels = [], []
        splits = split_mnist_by_input(inputs)
        for i, (img, lab) in enumerate(splits):
            client_data.append(img)
            client_labels.append(lab)

        X_train, Y_train = np.vstack(client_data), np.hstack(client_labels)




    #####################################################################
    #                        Predefined MNIST dataset
    #####################################################################
    elif dataset_name == 'mnist_5client5class_alpha0':
        data_class1, label_class1 = subset_byClass(digit_labels=[0], num_data=1000, seed=args.seed)
        data_class2, label_class2 = subset_byClass(digit_labels=[1], num_data=1000, seed=args.seed)
        data_class3, label_class3 = subset_byClass(digit_labels=[2], num_data=1000, seed=args.seed)
        data_class4, label_class4 = subset_byClass(digit_labels=[3], num_data=1000, seed=args.seed)
        data_class5, label_class5 = subset_byClass(digit_labels=[4], num_data=1000, seed=args.seed)

        client_images_c1, client_labels_c1 = split_dataset(data_class1, label_class1, [1000, 0, 0, 0, 0])
        client_images_c2, client_labels_c2 = split_dataset(data_class2, label_class2, [0, 1000, 0, 0, 0])
        client_images_c3, client_labels_c3 = split_dataset(data_class3, label_class3, [0, 0, 1000, 0, 0])
        client_images_c4, client_labels_c4 = split_dataset(data_class4, label_class4, [0, 0, 0, 1000, 0])
        client_images_c5, client_labels_c5 = split_dataset(data_class5, label_class5, [0, 0, 0, 0, 1000])

        client_data = [np.vstack([client_images_c1[0]]),
                       np.vstack([client_images_c2[1]]),
                       np.vstack([client_images_c3[2]]),
                       np.vstack([client_images_c4[3]]),
                       np.vstack([client_images_c5[4]])]

        client_labels = [np.hstack([client_labels_c1[0]]),
                         np.hstack([client_labels_c2[1]]),
                         np.hstack([client_labels_c3[2]]),
                         np.hstack([client_labels_c4[3]]),
                         np.hstack([client_labels_c5[4]])]

        X_train, Y_train = np.vstack([data_class1, data_class2, data_class3, data_class4, data_class5]), \
                           np.hstack([label_class1, label_class2, label_class3, label_class4, label_class5])



    elif dataset_name == 'mnist_5client10class_alpha0':
        data_class1, label_class1 = subset_byClass(digit_labels=[0], num_data=1000, seed=args.seed)
        data_class2, label_class2 = subset_byClass(digit_labels=[1], num_data=1000, seed=args.seed)
        data_class3, label_class3 = subset_byClass(digit_labels=[2], num_data=1000, seed=args.seed)
        data_class4, label_class4 = subset_byClass(digit_labels=[3], num_data=1000, seed=args.seed)
        data_class5, label_class5 = subset_byClass(digit_labels=[4], num_data=1000, seed=args.seed)
        data_class6, label_class6 = subset_byClass(digit_labels=[5], num_data=1000, seed=args.seed)
        data_class7, label_class7 = subset_byClass(digit_labels=[6], num_data=1000, seed=args.seed)
        data_class8, label_class8 = subset_byClass(digit_labels=[7], num_data=1000, seed=args.seed)
        data_class9, label_class9 = subset_byClass(digit_labels=[8], num_data=1000, seed=args.seed)
        data_class10, label_class10 = subset_byClass(digit_labels=[9], num_data=1000, seed=args.seed)

        client_images_c1, client_labels_c1 = split_dataset(data_class1, label_class1, [1000, 0, 0, 0, 0])
        client_images_c2, client_labels_c2 = split_dataset(data_class2, label_class2, [0, 1000, 0, 0, 0])
        client_images_c3, client_labels_c3 = split_dataset(data_class3, label_class3, [0, 0, 1000, 0, 0])
        client_images_c4, client_labels_c4 = split_dataset(data_class4, label_class4, [0, 0, 0, 1000, 0])
        client_images_c5, client_labels_c5 = split_dataset(data_class5, label_class5, [0, 0, 0, 0, 1000])
        client_images_c6, client_labels_c6 = split_dataset(data_class6, label_class6, [1000, 0, 0, 0, 0])
        client_images_c7, client_labels_c7 = split_dataset(data_class7, label_class7, [0, 1000, 0, 0, 0])
        client_images_c8, client_labels_c8 = split_dataset(data_class8, label_class8, [0, 0, 1000, 0, 0])
        client_images_c9, client_labels_c9 = split_dataset(data_class9, label_class9, [0, 0, 0, 1000, 0])
        client_images_c10, client_labels_c10 = split_dataset(data_class10, label_class10, [0, 0, 0, 0, 1000])

        client_data = [np.vstack([client_images_c1[0], client_images_c6[0]]),
                       np.vstack([client_images_c2[1], client_images_c7[1]]),
                       np.vstack([client_images_c3[2], client_images_c8[2]]),
                       np.vstack([client_images_c4[3], client_images_c9[3]]),
                       np.vstack([client_images_c5[4], client_images_c10[4]])]

        client_labels = [np.hstack([client_labels_c1[0], client_labels_c6[0]]),
                         np.hstack([client_labels_c2[1], client_labels_c7[1]]),
                         np.hstack([client_labels_c3[2], client_labels_c8[2]]),
                         np.hstack([client_labels_c4[3], client_labels_c9[3]]),
                         np.hstack([client_labels_c5[4], client_labels_c10[4]])]

        X_train, Y_train = np.vstack([data_class1, data_class2, data_class3, data_class4, data_class5,
                                      data_class6, data_class7, data_class8, data_class9, data_class10]), \
                           np.hstack([label_class1, label_class2, label_class3, label_class4, label_class5,
                                      label_class6, label_class7, label_class8, label_class9, label_class10])



    elif dataset_name == 'mnist_2class':
        # X_train, Y_train = subset_byClass(digit_labels=[2, 8], num_data=args.n_data, seed=args.seed)
        X_train, Y_train = subset_byClass(digit_labels=[0, 1], num_data=args.n_data, seed=args.seed)

        if not isCent:
            client_data, client_labels, dict_users = \
                mnist_subset_noniid(X_train, Y_train, num_users=args.n_users, method="dir", num_data=args.n_data * 2,
                                    alpha=args.alpha, seed=args.seed, path=folder_path)


    elif dataset_name == 'mnist_3class':
        X_train, Y_train = subset_byClass(digit_labels=[1, 6, 8], num_data=args.n_data, seed=args.seed)

        if not isCent:
            client_data, client_labels, dict_users = \
                mnist_subset_noniid(X_train, Y_train, num_users=args.n_users, method="dir", num_data=args.n_data * 3,
                                    alpha=args.alpha, seed=args.seed, path=folder_path)



    elif dataset_name == 'mnist_3class_iid_case':
        inputs = [
            {"class": [0, 1, 2], "num_samples": [500, 500, 500]},
            {"class": [0, 1, 2], "num_samples": [500, 500, 500]},
            {"class": [0, 1, 2], "num_samples": [500, 500, 500]},
        ]

        client_data, client_labels = [], []
        splits = split_mnist_by_input(inputs)
        for i, (img, lab) in enumerate(splits):
            client_data.append(img)
            client_labels.append(lab)

        X_train, Y_train = np.vstack(client_data), np.hstack(client_labels)




    elif dataset_name == 'mnist_3class_iid_case2':
        inputs = [
            {"class": [0, 1, 2], "num_samples": [1500, 500, 1700]},
            {"class": [0, 1, 2], "num_samples": [500, 1500, 300]},
        ]

        client_data, client_labels = [], []
        splits = split_mnist_by_input(inputs)
        for i, (img, lab) in enumerate(splits):
            client_data.append(img)
            client_labels.append(lab)

        X_train, Y_train = np.vstack(client_data), np.hstack(client_labels)





    elif dataset_name == 'mnist_5class':
        X_train, Y_train = subset_byClass(digit_labels=[0, 1, 2, 3, 4], num_data=args.n_data, seed=args.seed)

        if not isCent:
            if args.iid:
                client_data, client_labels, dict_users = \
                    mnist_subset_iid(X_train, Y_train, num_users=args.n_users, seed=args.seed, path=folder_path)
            else:
                client_data, client_labels, dict_users = \
                    mnist_subset_noniid(X_train, Y_train, num_users=args.n_users, method="dir",
                                        num_data=args.n_data * 5,
                                        alpha=args.alpha, seed=args.seed, path=folder_path)


    elif dataset_name == 'mnist_10class':
        X_train, Y_train = subset_byClass(digit_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], num_data=args.n_data,
                                          seed=args.seed)

        if not isCent:
            if args.iid:
                client_data, client_labels, dict_users = \
                    mnist_subset_iid(X_train, Y_train, num_users=args.n_users, seed=args.seed, path=folder_path)
            else:
                client_data, client_labels, dict_users = \
                    mnist_subset_noniid(X_train, Y_train, num_users=args.n_users, method="dir",
                                        num_data=args.n_data * 10,
                                        alpha=args.alpha, seed=args.seed, path=folder_path)


    elif dataset_name == 'mnist_3class3client_midNonIID_small':
        data_class1, label_class1 = subset_byClass(digit_labels=[1], num_data=300, seed=args.seed)
        data_class2, label_class2 = subset_byClass(digit_labels=[6], num_data=300, seed=args.seed)
        data_class3, label_class3 = subset_byClass(digit_labels=[8], num_data=300, seed=args.seed)

        client_images_c1, client_labels_c1 = split_dataset(data_class1, label_class1, [0, 250, 50])
        client_images_c2, client_labels_c2 = split_dataset(data_class2, label_class2, [50, 0, 250])
        client_images_c3, client_labels_c3 = split_dataset(data_class3, label_class3, [250, 50, 0])

        client_data = [np.vstack([client_images_c1[0], client_images_c2[0], client_images_c3[0]]),
                       np.vstack([client_images_c1[1], client_images_c2[1], client_images_c3[1]]),
                       np.vstack([client_images_c1[2], client_images_c2[2], client_images_c3[2]])]

        client_labels = [np.hstack([client_labels_c1[0], client_labels_c2[0], client_labels_c3[0]]),
                         np.hstack([client_labels_c1[1], client_labels_c2[1], client_labels_c3[1]]),
                         np.hstack([client_labels_c1[2], client_labels_c2[2], client_labels_c3[2]])]

        X_train, Y_train = np.vstack([data_class1, data_class2, data_class3]), np.hstack(
            [label_class1, label_class2, label_class3])



    elif dataset_name == 'mnist_3class3client_alpha0_small':
        data_class1, label_class1 = subset_byClass(digit_labels=[1], num_data=300, seed=args.seed)
        data_class2, label_class2 = subset_byClass(digit_labels=[6], num_data=300, seed=args.seed)
        data_class3, label_class3 = subset_byClass(digit_labels=[8], num_data=300, seed=args.seed)

        client_images_c1, client_labels_c1 = split_dataset(data_class1, label_class1, [300, 0, 0])
        client_images_c2, client_labels_c2 = split_dataset(data_class2, label_class2, [0, 300, 0])
        client_images_c3, client_labels_c3 = split_dataset(data_class3, label_class3, [0, 0, 300])

        client_data = [np.vstack([client_images_c1[0]]),
                       np.vstack([client_images_c2[1]]),
                       np.vstack([client_images_c3[2]])]

        client_labels = [np.hstack([client_labels_c1[0]]),
                         np.hstack([client_labels_c2[1]]),
                         np.hstack([client_labels_c3[2]])]

        X_train, Y_train = np.vstack([data_class1, data_class2, data_class3]), np.hstack(
            [label_class1, label_class2, label_class3])



    elif dataset_name == 'mnist_3class3client_nonIID_small':
        data_class1, label_class1 = subset_byClass(digit_labels=[1], num_data=300, seed=args.seed)
        data_class2, label_class2 = subset_byClass(digit_labels=[6], num_data=300, seed=args.seed)
        data_class3, label_class3 = subset_byClass(digit_labels=[8], num_data=300, seed=args.seed)

        client_images_c1, client_labels_c1 = split_dataset(data_class1, label_class1, [280, 10, 10])
        client_images_c2, client_labels_c2 = split_dataset(data_class2, label_class2, [10, 280, 10])
        client_images_c3, client_labels_c3 = split_dataset(data_class3, label_class3, [10, 10, 280])

        client_data = [np.vstack([client_images_c1[0], client_images_c2[0], client_images_c3[0]]),
                       np.vstack([client_images_c1[1], client_images_c2[1], client_images_c3[1]]),
                       np.vstack([client_images_c1[2], client_images_c2[2], client_images_c3[2]])]

        client_labels = [np.hstack([client_labels_c1[0], client_labels_c2[0], client_labels_c3[0]]),
                         np.hstack([client_labels_c1[1], client_labels_c2[1], client_labels_c3[1]]),
                         np.hstack([client_labels_c1[2], client_labels_c2[2], client_labels_c3[2]])]

        X_train, Y_train = np.vstack([data_class1, data_class2, data_class3]), np.hstack(
            [label_class1, label_class2, label_class3])



    elif dataset_name == 'mnist_3class3client_IID_small':
        data_class1, label_class1 = subset_byClass(digit_labels=[1], num_data=300, seed=args.seed)
        data_class2, label_class2 = subset_byClass(digit_labels=[6], num_data=300, seed=args.seed)
        data_class3, label_class3 = subset_byClass(digit_labels=[8], num_data=300, seed=args.seed)

        client_images_c1, client_labels_c1 = split_dataset(data_class1, label_class1, [100, 100, 100])
        client_images_c2, client_labels_c2 = split_dataset(data_class2, label_class2, [100, 100, 100])
        client_images_c3, client_labels_c3 = split_dataset(data_class3, label_class3, [100, 100, 100])

        client_data = [np.vstack([client_images_c1[0], client_images_c2[0], client_images_c3[0]]),
                       np.vstack([client_images_c1[1], client_images_c2[1], client_images_c3[1]]),
                       np.vstack([client_images_c1[2], client_images_c2[2], client_images_c3[2]])]

        client_labels = [np.hstack([client_labels_c1[0], client_labels_c2[0], client_labels_c3[0]]),
                         np.hstack([client_labels_c1[1], client_labels_c2[1], client_labels_c3[1]]),
                         np.hstack([client_labels_c1[2], client_labels_c2[2], client_labels_c3[2]])]

        X_train, Y_train = np.vstack([data_class1, data_class2, data_class3]), np.hstack(
            [label_class1, label_class2, label_class3])



    else:
        print(f'Dataset {dataset_name} Not implemented yet...')
        sys.exit(0)

    if isCent:
        return X_train, Y_train
    return X_train, Y_train, client_data, client_labels, dict_users


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.iid = True
    args.n_users = 3
    args.seed = 42
    args.alpha = 0.5

    dataset_name = 'mnist_3class3client_alpha0.5'

    X_train, Y_train, client_data, client_labels, dict_users = get_mnist_dataset(args, dataset_name,
                                                                                 folder_path='test', isCent=False)


