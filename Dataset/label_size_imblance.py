import numpy as np
import torchvision
import os
import random
from Dataset.mnist_datasets import split_mnist_by_input, split_fashmnist_by_input, split_rnaseq_by_input
from Dataset.rnaseq_datasets import load_MACOSKO
from Dataset.cifar import load_cifar10, load_cifar100
from Dataset.cifar_datasets import split_cifar10_by_input, split_cifar100_by_input



def split_samples_over_class(n_samples, n_sites, seed=42):
    random.seed(seed)
    size = int(np.ceil(n_samples / n_sites))
    num = n_samples // size
    n_remains = n_samples - num * size
    if n_remains == 0:
        sizes_set = [size] * num
    else:
        sizes_set = [size] * num + [n_remains]
    random.shuffle(sizes_set)
    return sizes_set



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



def split_mnist_dataset(args, folder_path):
    n_clients = args.n_users
    ccc = args.n_classes

    np.random.seed(args.seed)

    # Load MNIST dataset
    train_images, train_labels = load_mnist()
    train_images = train_images.reshape(train_images.shape[0], -1)

    data = train_images  # (60000,784)
    labels = train_labels

    n_data = len(data)
    n_classes = len(np.unique(labels))

    # Assign classes to clients:
    client_classes = {}
    class_set = []

    # Initialize a count array to keep track of how many times each class is selected
    class_counts = np.zeros(n_classes)
    
    
    ''' Fake client_classes '''
    startC = 0

    for i in range(n_clients):
        
        if i < startC:
            continue
      
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
    splits = split_mnist_by_input(inputs)

    for i, (img, lab) in enumerate(splits):
        clients_data.append(img)
        clients_labels.append(lab)

    # Write clients data summary to a text file
    output_file = f"MNIST_data{n_data}_u{n_clients}_c{ccc}_seed{args.seed}.txt"
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





def split_fashmnist_dataset(args, folder_path):
    n_clients = args.n_users
    ccc = args.n_classes

    np.random.seed(args.seed)

    # Load FMNIST dataset
    train_images, train_labels = load_fashmnist()
    train_images = train_images.reshape(train_images.shape[0], -1)

    data = train_images  # (60000,784)
    labels = train_labels

    n_data = len(data)
    n_classes = len(np.unique(labels))

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
    splits = split_fashmnist_by_input(inputs)

    for i, (img, lab) in enumerate(splits):
        clients_data.append(img)
        clients_labels.append(lab)

    # Write clients data summary to a text file
    output_file = f"FashMNIST_data{n_data}_u{n_clients}_c{ccc}_seed{args.seed}.txt"
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







def split_Kuzushiji49_dataset(args, folder_path):
    n_clients = args.n_users
    ccc = args.n_classes

    np.random.seed(args.seed)

    # Load Kuzushiji49 dataset
    train_images, train_labels = load_Kuzushiji49()
    train_images = train_images.reshape(train_images.shape[0], -1)

    data = train_images
    labels = train_labels

    n_data = len(data)
    n_classes = len(np.unique(labels))

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
    splits = split_fashmnist_by_input(inputs)

    for i, (img, lab) in enumerate(splits):
        clients_data.append(img)
        clients_labels.append(lab)

    # Write clients data summary to a text file
    output_file = f"Kuzushiji49_data{n_data}_u{n_clients}_c{ccc}_seed{args.seed}.txt"
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






def split_macosko_dataset(args, folder_path):
    n_clients = args.n_users
    ccc = args.n_classes

    np.random.seed(args.seed)

    # Load mouse dataset
    train_images, _, train_labels, _ = load_MACOSKO()
    train_images = train_images.reshape(train_images.shape[0], -1)
    print(train_images.shape)

    data = train_images
    labels = train_labels

    n_data = len(data)
    n_classes = len(np.unique(labels))

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
    output_file = f"Macosko_data{n_data}_u{n_clients}_c{ccc}_seed{args.seed}.txt"
    n_sum = 0
    with open(os.path.join(folder_path, output_file), "w") as file:
        for i in range(n_clients):
            client_data, client_labels = clients_data[i], clients_labels[i]

            # Count occurrences of each digit label
            label_counts = [np.sum(client_labels == j) for j in range(n_classes)]
            sum_labels = np.sum(label_counts)

            # Formatting and writing to file
            label_counts_str = ' '.join(str(count) for count in label_counts)
            line = f"User {i}: {label_counts_str} sum: {sum_labels}\n"
            file.write(line)
            n_sum += sum_labels

    assert n_sum == n_data

    return data, labels, clients_data, clients_labels







def split_cifar10_dataset(args, folder_path):
    n_clients = args.n_users
    ccc = args.n_classes

    np.random.seed(args.seed)

    # Load cifar10 dataset
    train_images, train_labels = load_cifar10()
    train_images = train_images.reshape(train_images.shape[0], -1)

    data = np.array(train_images)  # (50000,784)
    labels = train_labels

    n_data = len(data)
    n_classes = len(np.unique(labels)) 

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
    splits = split_cifar10_by_input(inputs)

    for i, (img, lab) in enumerate(splits):
        clients_data.append(img)
        clients_labels.append(lab)

    # Write clients data summary to a text file
    output_file = f"CIFAR10_data{n_data}_u{n_clients}_c{ccc}_seed{args.seed}.txt"
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




def valid_3permutations(lst):
    length = len(lst)
    perm1, perm2, perm3 = lst[:], lst[:], lst[:]
    random.shuffle(perm1)
    random.shuffle(perm2)
    random.shuffle(perm3)

    # Check for any index having the same value across permutations and re-shuffle if found
    while any(perm1[i] == perm2[i] or perm2[i] == perm3[i] or perm1[i] == perm3[i] for i in range(length)):
        random.shuffle(perm2)
        random.shuffle(perm3)
    return perm1, perm2, perm3




def split_cifar100_dataset(args, folder_path):
    n_clients = args.n_users
    ccc = args.n_classes

    np.random.seed(args.seed)

    # Load cifar10 dataset
    train_images, train_labels = load_cifar100()
    train_images = train_images.reshape(train_images.shape[0], -1)

    data = train_images  # (50000,784)
    labels = train_labels

    n_data = len(data)
    n_classes = len(np.unique(labels))

    # Assign classes to clients:
    client_classes = {}
    class_set = []

    # Initialize a count array to keep track of how many times each class is selected
    class_counts = np.zeros(n_classes)

    if n_clients == n_classes:
        indices1, indices2, indices3 = valid_3permutations([*range(n_classes)])

        if ccc == 2:
            for i in range(n_clients): 
                selected_classes = [indices1[i], indices2[i]] 
                client_classes[i] = selected_classes
                class_set += list(selected_classes)

        elif ccc == 3:
            for i in range(n_clients): 
                selected_classes = [indices1[i], indices2[i], indices3[i]] 
                client_classes[i] = selected_classes
                class_set += list(selected_classes)

        else:
            print('wrong ccc')
            
    else:
        print('n_clients != n_classes')
        exit(0)


    # for i in range(n_clients):
    #     available_classes = list(range(n_classes))
    #     selected_classes = []

    #     while len(selected_classes) < ccc:
    #         # Weight classes by their inverse frequency (less frequent classes are more likely to be chosen)
    #         weights = 1 / (class_counts[available_classes] + 1)
    #         weights /= weights.sum()

    #         # Randomly select one class based on the weights
    #         selected_class = np.random.choice(available_classes, p=weights)
    #         selected_classes.append(selected_class)

    #         # Update class counts and remove the selected class from available classes
    #         class_counts[selected_class] += 1
    #         available_classes.remove(selected_class)

    #     client_classes[i] = selected_classes
    #     class_set += list(selected_classes)


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
    splits = split_cifar100_by_input(inputs)

    for i, (img, lab) in enumerate(splits):
        clients_data.append(img)
        clients_labels.append(lab)

    # Write clients data summary to a text file
    output_file = f"CIFAR100_data{n_data}_u{n_clients}_c{ccc}_seed{args.seed}.txt"
    n_sum = 0
    with open(os.path.join(folder_path, output_file), "w") as file:
        for i in range(n_clients):
            client_data, client_labels = clients_data[i], clients_labels[i]

            # Count occurrences of each digit label
            label_counts = [np.sum(client_labels == j) for j in range(100)]
            sum_labels = np.sum(label_counts)

            # Formatting and writing to file
            label_counts_str = ' '.join(str(count) for count in label_counts)
            line = f"User {i}: {label_counts_str} sum: {sum_labels}\n"
            file.write(line)
            n_sum += sum_labels

    assert n_sum == n_data

    return data, labels, clients_data, clients_labels




