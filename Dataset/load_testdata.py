import numpy as np
import random
import copy
import os
import torchvision 

from Dataset.rnaseq_datasets import load_MACOSKO
from Dataset.cifar import load_test_cifar10, load_test_cifar100 



def load_testdata(dataset):

    if "fashmnist" in dataset:
        fmnist_test = torchvision.datasets.FashionMNIST('data', train=False, download=True, transform=None)
        x_test, y_test = fmnist_test.data.float().numpy(), fmnist_test.targets
        x_test = x_test.reshape(x_test.shape[0], -1)
        x_test = x_test / 255.


    elif "mnist" in dataset and "fashmnist" not in dataset: 
        mnist_test = torchvision.datasets.MNIST('data', train=False, download=True, transform=None)
        x_test, y_test = mnist_test.data.float().numpy(), mnist_test.targets
        x_test = x_test.reshape(x_test.shape[0], -1)
        x_test = x_test / 255.


    elif 'rnaseq' in dataset:
        _, x_test, _, y_test = load_MACOSKO()


    elif 'cifar10' in dataset and 'cifar100' not in dataset:
        x_test, y_test = load_test_cifar10()


    elif 'cifar100' in dataset:
        x_test, y_test = load_test_cifar100()


    return x_test, y_test



