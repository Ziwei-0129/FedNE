import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
import os
import torchvision





def fmnist_case1_dataset(args, folder_path):

    # load F-MNIST
    fmnist_train = torchvision.datasets.FashionMNIST('data', train=True, download=True, transform=None)
    x_train, y_train = fmnist_train.data.float().numpy(), fmnist_train.targets
    train_images = x_train / 255.
    train_images = train_images.reshape(train_images.shape[0], -1)
    train_labels = y_train

    # Class mappings
    class_map = {
        "T-shirt/top": 0,
        "Trouser": 1,
        "Pullover": 2,
        "Dress": 3,
        "Coat": 4,
        "Sandal": 5,
        "Shirt": 6,
        "Sneaker": 7,
        "Bag": 8,
        "Ankle boot": 9,
    }
    
    reverse_map = {v: k for k, v in class_map.items()}

    # Define client data needs
    client_classes = {
        'C1': ["Sandal"],                               # 0
        'C2': ["Sandal", "Sneaker", "Ankle boot"],      # 1
        'C3': ["Sneaker", "Ankle boot"],                # 2
        'C4': ["T-shirt/top", "Shirt", "Pullover"],     # 3
        'C5': ["T-shirt/top", "Shirt"],                 # 4
        'C6': ["Dress", "Trouser"],                     # 5
        'C7': ["Dress"],                                # 6
        'C8': ["Trouser"],                              # 7
        'C9': ["Pullover", "Coat"],                     # 8
        'C10': ["Bag"],                                 # 9
    }

    # Data structures for client data
    client_data = {client: [] for client in client_classes}
    client_labels = {client: [] for client in client_classes}

    # Split data for each class
    for cls in class_map:
        indices = [i for i, label in enumerate(train_labels) if label == class_map[cls]]
        np.random.shuffle(indices)
        split_size = len(indices) // len([client for client in client_classes if cls in client_classes[client]])
        start = 0

        # Distribute class data to clients
        for client in client_classes:
            if cls in client_classes[client]: 
                end = start + split_size
                client_data[client].extend(train_images[indices[start:end]])
                client_labels[client].extend([class_map[cls]] * (end - start))
                start = end


    # Prepare counts per class for each client
    client_counts = {}
    for client, labels in client_labels.items():
        counts = [0] * 10  # For 10 classes
        for label in labels:
            counts[label] += 1
        client_counts[client] = counts
    

    # Write clients data summary to a text file
    output_file = f"FMNIST_u{10}_case1_seed{args.seed}.txt"
    with open(os.path.join(folder_path, output_file), 'w') as f:
        # Write header with class names
        f.write("Class Names: " + " ".join([reverse_map[i] for i in range(10)]) + '\n')
        # Write data for each client
        for client, counts in client_counts.items():
            line = f"User {client[1:]}: " + " ".join(map(str, counts)) + f" sum: {sum(counts)}\n"
            f.write(line)
            
    client_data = [np.array(client_data[c]) for c in client_data.keys()]
    client_labels = [client_labels[c] for c in client_labels.keys()]
    
    return train_images, train_labels, client_data, client_labels


