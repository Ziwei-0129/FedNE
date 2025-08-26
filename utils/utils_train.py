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

# from numpy.core._exceptions import _UFuncNoLoopError
# pickle.dumps(_UFuncNoLoopError)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




def plt_global(z_umap, labels=None, name=None):
    import matplotlib.colors as mcolors

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



def plt_global_wClientLabels(folder_path, round_id, encoder_glob, client_datas, client_labels, name):
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(12, 12))

    cmap = plt.get_cmap("tab10")
    norm = mcolors.Normalize(vmin=0, vmax=9)

    Z_list = []

    '''global'''
    for c, data in enumerate(client_datas):
        labels = client_labels[c]
        Z_c = encoder_glob(torch.Tensor(data).cuda()).detach().cpu()
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




def test_global(encoder, data, n_nbrs, test_bs, seed, graph, loss_aggregation='mean'):
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

    loss, loss_pos, loss_neg = embedder.test(data, graph=graph, batch_size=test_bs, seed=seed, loss_aggregation=loss_aggregation)
    return loss, loss_pos, loss_neg



def clientUpdate(
        isSurrogate,
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
):

    if client_funct_dict[1] is not None and client_funct_dict[1][0] is not None:
        for key in client_funct_dict[1]:
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


    embedding, mean_att_loss = \
            embedder.fit_transform(local_data, local_labels, isSurrogate, 
                                client_graph_info, client_ratios, client_funct_dict, 
                                client_attraction_thred, isCent, add_both)

    return embedder, embedding, mean_att_loss


