import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cne
import copy
import torch.nn as nn
import math
import pickle
import random

# from src.utils import *
# from src.clustering import *

from numpy.core._exceptions import _UFuncNoLoopError
pickle.dumps(_UFuncNoLoopError)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




def plt_global(z_umap, labels=None, name=None):
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap

    if len(np.unique(labels)) == 100: 
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(
            z_umap[:, 0],
            z_umap[:, 1],
            c=np.arange(100),
            # cmap=cmap,
            s=8,
            alpha=1, 
            label=colors,
        )
        ax.axis('equal') 
        plt.savefig(name, bbox_inches='tight')
        plt.clf()
        plt.close()

    else:
        if labels is None:
            colors = 'gray'
            cmap = None
        else:
            colors = np.array(labels).astype(int)
            if len(np.unique(labels)) == 10:
                cmap = 'tab10' 
            else:
                cmap = 'Set1'

        cmap = plt.get_cmap("tab10")
        norm = mcolors.Normalize(vmin=0, vmax=9)
        colors = cmap(norm(np.array(labels).astype(int)))

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.scatter(
            z_umap[:, 0],
            z_umap[:, 1],
            c=colors,
            cmap=cmap,
            s=8,
            alpha=1,
            # rasterized=True,
            label=colors,
        )
        ax.axis('equal')
        # plt.legend()
        plt.savefig(name, bbox_inches='tight')
        plt.clf()
        plt.close()

    return z_umap



def plt_global_wClientLabels(folder_path, round_id, encoder_glob, client_datas, client_labels, name, device):
    import matplotlib.colors as mcolors
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(12, 12))

    if len(client_labels) == 100: 
        Z_list = []

        # Generate 100 random colors
        np.random.seed(0)  
        colors = np.random.rand(100, 3)  
        cmap = ListedColormap(colors)
        norm = mcolors.Normalize(vmin=0, vmax=99)

        '''global'''
        for c, data in enumerate(client_datas):
            labels = client_labels[c]
            Z_c = encoder_glob(torch.Tensor(data).to(device)).detach().cpu()
            colors = cmap(norm(np.array(labels).astype(int)))
            Z_list.append(Z_c)

            ax.scatter(
                Z_c[:, 0],
                Z_c[:, 1],
                c=colors,
                cmap=cmap,
                s=8,
                alpha=1,
                label=colors,
                marker='.',
            )
        ax.axis('equal')
        plt.savefig(name, bbox_inches='tight', dpi=80)
        plt.clf()
        plt.close()

    else:
        cmap = plt.get_cmap("tab10")
        norm = mcolors.Normalize(vmin=0, vmax=9)

        Z_list = []

        '''global'''
        for c, data in enumerate(client_datas):
            labels = client_labels[c]
            Z_c = encoder_glob(torch.Tensor(data).to(device)).detach().cpu()
            Z_list.append(Z_c)

            colors = cmap(norm(np.array(labels).astype(int)))
            ax.scatter(
                Z_c[:, 0],
                Z_c[:, 1],
                c=colors,
                cmap=cmap,
                s=8,
                alpha=1,
                label=colors,
                marker='.',
            )
        ax.axis('equal')
        plt.savefig(name, bbox_inches='tight', dpi=80)
        plt.clf()
        plt.close()


    """
    # '''individual'''
    if not os.path.exists(os.path.join(folder_path, 'local_views')):
        os.makedirs(os.path.join(folder_path, 'local_views'))

    for c, z_data in enumerate(Z_list):
        fig, ax = plt.subplots(figsize=(12, 12))
        labels = client_labels[c]
        colors = cmap(norm(np.array(labels).astype(int)))
        ax.scatter(
            z_data[:, 0],
            z_data[:, 1],
            c=colors,
            cmap=cmap,
            s=8,
            alpha=1,
            label=colors,
            marker='.',
        )
        ax.axis('equal')
        plt.savefig(os.path.join(folder_path, 'local_views', f"R{round_id}_C{c}.png"), bbox_inches='tight', dpi=50)
        plt.clf()
        plt.close()
    """



def split_row_col(num_users):
    if num_users == 2:
        return 1, 2
    if num_users == 4:
        return 2, 2
    col_size = 3
    num_rows = num_users // col_size
    if num_users % col_size > 0:
        num_rows += 1
    return num_rows, col_size



def FedAvg(w, ratios=None):
    """ Returns the average of the weights. """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * ratios[0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * ratios[i]
    return w_avg




def modelUpdate(
    train_data,
    model,
    k,
    lr,
    epochs_local,
    batch_size,
    n_batches,
    client_graph_info,
    client_funct_dict=None,
    isCent=True,
):
    embedder = cne.CNE(
            CF_set=None,
            n_clients=None,
            anneal_lr="none",
            batch_size=batch_size,
            n_epochs=epochs_local,
            n_batches=n_batches,
            model=model,
            loss_mode="neg",
            k=k,
            optimizer="adam",
            learning_rate=lr,
            momentum=0.0,
            parametric=True,
            print_freq_epoch=1,
            on_gpu=True,
            client_id=None,)

    embedding, mean_att_loss = embedder.fit_transform(train_data, None, None, None, None, 
                            client_graph_info, None, client_funct_dict, isCent)

    return embedder, embedding, mean_att_loss




def test_global(encoder, data, n_nbrs, test_bs, seed, graph, device, loss_aggregation='mean'):
    encoder.eval()

    ''' create network '''
    embd_layer = torch.nn.Embedding.from_pretrained(torch.tensor(data), freeze=True)
    model = torch.nn.Sequential(embd_layer, encoder)

    embedder = cne.CNE(
                model=model,
                loss_mode="neg",
                k=n_nbrs,
                optimizer="adam",
                anneal_lr=False,
                momentum=0.0,
                parametric=True,
                n_epochs=-1,
                print_freq_epoch=1,
                on_gpu=True,)

    loss, loss_pos, loss_neg = embedder.test(data, graph=graph, batch_size=test_bs, 
                                seed=seed, loss_aggregation=loss_aggregation, device=device)
    return loss, loss_pos, loss_neg



def clientUpdate(
        isSurrogate, isLocalTrain, isLowerbound, isFakeDecent,
        client_ratios,
        client_id,
        local_data, 
        local_labels,
        n_clients,
        encoder,
        k,
        lr,
        epochs_local,
        batch_size,
        n_batches,
        client_graph_info,
        client_funct_dict,
        isCent=False,
        add_both=True,
        client_attraction_thred=None,
        optim_state=None,
):

    if client_funct_dict[1] is not None and client_funct_dict[1][0] is not None:
        for key in client_funct_dict[0]:
            if add_both and client_funct_dict[0][key] is not None:
                client_funct_dict[0][key].eval()
            client_funct_dict[1][key].eval()

  

    embd_layer = torch.nn.Embedding.from_pretrained(torch.tensor(local_data), freeze=True)
    model = torch.nn.Sequential(embd_layer, encoder)


    embedder = cne.CNE(
            CF_set=None,
            n_clients=n_clients,
            anneal_lr="none",
            batch_size=batch_size,
            n_epochs=epochs_local,
            n_batches=n_batches,
            model=model,
            loss_mode="neg",
            k=k,
            optimizer="adam",
            learning_rate=lr,
            momentum=0.0,
            parametric=True,
            print_freq_epoch=1,
            on_gpu=True,
            client_id=client_id,
    )


    embedding, mean_att_loss, optim_state = \
            embedder.fit_transform(local_data, local_labels, isSurrogate, isLocalTrain, isLowerbound, isFakeDecent,
                    client_graph_info, client_ratios, client_funct_dict, client_attraction_thred, isCent, add_both, optim_state)

    return embedder, embedding, mean_att_loss, optim_state





def get_client_similarities(client_sets, th):
    data_per_class = {}
    U_per_class = {}
    K = 3

    # -------------------------- dataset --------------------------
    for i, client_data in enumerate(client_sets):

        length = len(client_data)
        data_per_class[i] = client_data.reshape(length, -1).T.astype(float)

        u1_temp, sh1_temp, vh1_temp = np.linalg.svd(data_per_class[i], full_matrices=False)
        u1_temp = u1_temp / np.linalg.norm(u1_temp, ord=2, axis=0)
        U_per_class[i] = u1_temp[:, 0:K]

    # -------------------------- subspace angles --------------------------
    num = len(client_sets)

    sim_angle_min = np.zeros([num, num])
    sim_angle_tr = np.zeros([num, num])

    for i in range(num):
        for j in range(num):

            F, G = Eq_Basis(U_per_class[i], U_per_class[j])
            F_in_G = np.clip(F.T@G, a_min=-1, a_max=+1)

            Angle = np.arccos(np.abs(F_in_G))
            sim_angle_min[i,j] = (180/np.pi)*np.min(Angle)
            sim_angle_tr[i,j] = (180/np.pi)*np.trace(Angle)

 

    # -------------------------- hierarchical clustering --------------------------
    client_similarity = {c: [] for c in range(num)}
    # th = 10
    clusters = hierarchical_clustering(copy.deepcopy(sim_angle_min), thresh=th, linkage='average')
    # print(clusters)
    for group in clusters:
        for cid in group:
            client_similarity[cid] = [idx for idx in group if idx != cid]

    return client_similarity, clusters




def hierarchical_clustering(A, thresh=1.5, linkage='maximum'):
    '''
    Hierarchical Clustering Algorithm. It is based on single linkage, finds the minimum element and merges
    rows and columns replacing the minimum elements. It is working on adjacency matrix.

    :param: A (adjacency matrix), thresh (stopping threshold)
    :type: A (np.array), thresh (int)

    :return: clusters
    '''
    label_assg = {i: i for i in range(A.shape[0])}

    B = copy.deepcopy(A)
    step = 0
    while A.shape[0] > 1:
        np.fill_diagonal(A, -np.NINF)
        # print(f'step {step} \n {A}')
        step += 1
        ind = np.unravel_index(np.argmin(A, axis=None), A.shape)

        if A[ind[0], ind[1]] > thresh:
            print('Breaking HC')
            # print(f'A {B}')
            break
        else:
            np.fill_diagonal(A, 0)
            if linkage == 'maximum':
                Z = np.maximum(A[:, ind[0]], A[:, ind[1]])
            elif linkage == 'minimum':
                Z = np.minimum(A[:, ind[0]], A[:, ind[1]])
            elif linkage == 'average':
                Z = (A[:, ind[0]] + A[:, ind[1]]) / 2

            A[:, ind[0]] = Z
            A[:, ind[1]] = Z
            A[ind[0], :] = Z
            A[ind[1], :] = Z
            A = np.delete(A, (ind[1]), axis=0)
            A = np.delete(A, (ind[1]), axis=1)

            B = copy.deepcopy(A)
            if type(label_assg[ind[0]]) == list:
                label_assg[ind[0]].append(label_assg[ind[1]])
            else:
                label_assg[ind[0]] = [label_assg[ind[0]], label_assg[ind[1]]]

            label_assg.pop(ind[1], None)

            temp = []
            for k, v in label_assg.items():
                if k > ind[1]:
                    kk = k - 1
                    vv = v
                else:
                    kk = k
                    vv = v
                temp.append((kk, vv))

            label_assg = dict(temp)

    clusters = []
    for k in label_assg.keys():
        if type(label_assg[k]) == list:
            clusters.append(list(flatten(label_assg[k])))
        elif type(label_assg[k]) == int:
            clusters.append([label_assg[k]])

    return clusters




def Eq_Basis(A,B):
    AB=np.arccos(A.T@B)
    A_E=np.zeros((A.shape[0],A.shape[1]))
    B_E=np.zeros((B.shape[0],B.shape[1]))
    for i in range(AB.shape[0]):
        ind = np.unravel_index(np.argmin(AB, axis=None), AB.shape)
        AB[ind[0],:]=AB[:,ind[1]]=0
        A_E[:,i]=A[:,ind[0]]
        B_E[:,i]=B[:,ind[1]]
    return A_E, B_E




