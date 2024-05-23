import numpy as np
import torchvision
import os
import random
from Dataset.mnist_datasets import split_mnist_by_input, split_fashmnist_by_input



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



def load_Kuzushiji49():
    image_trn_path = 'kmnist/k49-train-imgs.npz'
    label_trn_path = 'kmnist/k49-train-labels.npz'
    image_tst_path = 'kmnist/k49-test-imgs.npz'
    label_tst_path = 'kmnist/k49-test-labels.npz'
    
    arr_image_trn = np.load(image_trn_path)['arr_0']
    arr_label_trn = np.load(label_trn_path)['arr_0'] 
    arr_image_tst = np.load(image_tst_path)['arr_0']
    arr_label_tst = np.load(label_tst_path)['arr_0']     
    
    x_train = arr_image_trn
    y_train = arr_label_trn
    x_train = x_train / 255.
    print(x_train.shape, y_train.shape)
    return x_train, y_train





def split_mnist_by_input(inputs):

    x_train, y_train = load_Kuzushiji49()
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




