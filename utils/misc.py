from scipy import sparse
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
import numpy as np
import os



def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True




def create_folder(args, isCent=False):
    folder_name = ""

    if args.n_data != -1:
        dataset_name = f"{args.dataset}_{args.n_data}"
    else:
        dataset_name = f"{args.dataset}"

    if isCent:
        folder_name = f'{dataset_name}_k{args.k}_e{args.epochs}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}_seed{args.seed}'

    else:

        if args.lowerbound:
            args.path = os.path.join(args.path, "Lowerbound")
            if not os.path.exists(args.path):
                os.makedirs(args.path)

            if args.iid == False and args.alpha is None:
                folder_name = f'{dataset_name}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}'
            elif args.iid:
                folder_name = f'{dataset_name}_iid{args.iid}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}'
            elif args.iid == False and args.alpha is not None:
                folder_name = f'{dataset_name}_iid{args.iid}_alpha{args.alpha}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}'
            else:
                print("Wrong Fed dataset...")
                exit(0)

        elif args.fakedecent:
            args.path = os.path.join(args.path, "Upperbound")
            if not os.path.exists(args.path):
                os.makedirs(args.path)

            if args.iid == False and args.alpha is None:
                folder_name = f'{dataset_name}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}'
            elif args.iid:
                folder_name = f'{dataset_name}_iid{args.iid}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}'
            elif args.iid == False and args.alpha is not None:
                folder_name = f'{dataset_name}_iid{args.iid}_alpha{args.alpha}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}'
            else:
                print("Wrong Fed dataset...")
                exit(0)

        elif args.local_train:
            print('???')


        elif args.surrogate:

            if args.start_round == 0:
                args.path = os.path.join(args.path, "Surrogate")
            else:
                args.path = os.path.join(args.path, "FedAvg")
            
            if not os.path.exists(args.path):
                os.makedirs(args.path)


            if args.iid:
                folder_name = f'{dataset_name}_iid{args.iid}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}_surrogate_startR{args.start_round}_seed{args.seed}'#_rWait{args.r_wait}'
            elif args.iid is False:
                folder_name = f'{dataset_name}_iid{args.iid}_c{args.n_classes}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}_surrogate_startR{args.start_round}_seed{args.seed}'#_rWait{args.r_wait}'
            else:
                print("Wrong Fed dataset...")
                exit(0)

        else:
            print(("Failed to create folder..."))
            exit(0)


    folder_name = folder_name.replace('_None', '')

    PATH = os.path.join(args.path, folder_name)
    if not os.path.exists(PATH):
        os.mkdir(PATH)

    return PATH





def scale_globdata(glob_dataset, scale):
    q_inds = np.random.choice([i for i in range(glob_dataset.shape[0])], int(glob_dataset.shape[0]*scale), replace=False)
    q_train = np.take(glob_dataset, q_inds, axis=0)
    return q_train



