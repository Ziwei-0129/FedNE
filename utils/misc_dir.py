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




def create_folder(args, isSurr=False):
    folder_name = ""
    
    if isSurr:
        args.path = os.path.join(args.path, "Surrogate")
        if not os.path.exists(args.path):
            os.makedirs(args.path)
        
        if args.iid:
            folder_name = f'{args.dataset}_iid{args.iid}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}_stepsize{args.step_size}_beta{args.beta}_seed{args.seed}' 
        elif args.iid is False and args.alpha is not None:
            folder_name = f'{args.dataset}_iid{args.iid}_alpha{args.alpha}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}_stepsize{args.step_size}_beta{args.beta}_seed{args.seed}'
        else:
            print("Wrong Fed dataset...")
            exit(0)
        
    else:
        args.path = os.path.join(args.path, "FedAvg")
        if not os.path.exists(args.path):
            os.makedirs(args.path)
            
        if args.iid:
            folder_name = f'{args.dataset}_iid{args.iid}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}_seed{args.seed}' 
        elif args.iid is False and args.alpha is not None:
            folder_name = f'{args.dataset}_iid{args.iid}_alpha{args.alpha}_k{args.k}_r{args.rounds}_u{args.n_users}_ep{args.epochs_local}_bs{args.batch_size}_nBatches{args.n_batches}_lr{args.lr}_seed{args.seed}'
        else:
            print("Wrong Fed dataset...")
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



