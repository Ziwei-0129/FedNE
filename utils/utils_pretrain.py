import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from annoy import AnnoyIndex
import copy
import torch
import torch.nn as nn
import os

from utils.utils_neigh import build_kNN_graph, neighborhood_cent
from utils.utils_train import modelUpdate, test_global



def sample_multivariate_gaussian(mean_clients, cov_clients, n_samples_list):
    new_samples = []
    for i, cid in enumerate(mean_clients.keys()):
        size = n_samples_list[i]
        mean_, cov_ = mean_clients[cid].squeeze(), cov_clients[cid].squeeze()
        X = np.random.multivariate_normal(mean_, cov_, size=size)
        new_image = np.clip(X, 0, 1)
        new_samples.append(new_image.astype(np.float32))
    return np.vstack(new_samples)



def test_and_plot(args, ep, encoder, images, labels, kNN_graph_glob):
    if labels is None:
        colors = 'gray'
        cmap = None
    else:
        colors = np.array(labels).astype(int)
        if len(np.unique(labels)) == 10:
            cmap = 'tab10'
        else:
            cmap = 'Set1'

    model_cne = encoder.eval().to('cuda')   #copy.deepcopy(embedder.model[1]).eval().to('cuda')
    z_cne_pretrain = model_cne(torch.tensor(images, device='cuda')).detach().cpu().numpy()

    plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
    plt.scatter(
        z_cne_pretrain[:, 0],
        z_cne_pretrain[:, 1],
        c=colors,
        cmap=cmap,
        s=8,
        alpha=1,
        rasterized=True,
    )
    plt.axis('equal')
    plt.savefig(os.path.join(args.folder_path, f"pretrain_e{ep}.png"), bbox_inches='tight')
    plt.clf()

    # Record losses
    test_loss, test_loss_pos, test_loss_neg = \
        test_global(copy.deepcopy(model_cne), images, args.k, args.test_bs, seed=args.seed, graph=kNN_graph_glob)
    return test_loss, test_loss_pos, test_loss_neg, z_cne_pretrain



def cne_pretrain(
        args,
        encoder,
        epochs,
        batch_size,
        n_batches,
        lr,
        mean_clients,
        cov_clients,
        client_size_list,
        globimages,
        globlabels,
        kNN_graph_glob,
    ):
    # from Server view
    n_total = sum(client_size_list)
    assert len(mean_clients) == len(cov_clients)

    # ========================= Sample syn global data =========================
    samples_glob = sample_multivariate_gaussian(mean_clients, cov_clients, n_samples_list=client_size_list)

    # ========================= Build global kNN graph =========================
    _, graph_glob_pretrain, __ = build_kNN_graph(dataset=samples_glob, n_nbrs=args.k)

    inds_pos_dict_pretrain, inds_neg_dict_pretrain, neigh_counts_dict_pretrain = \
        neighborhood_cent(kNN_graph=graph_glob_pretrain)

    neighborhood_pretrain = {
        'inds_pos_dict': inds_pos_dict_pretrain,
        'inds_neg_dict': inds_neg_dict_pretrain,
        'neigh_counts_dict': neigh_counts_dict_pretrain,
    }

    # Initialize CNE model
    embd_layer = torch.nn.Embedding.from_pretrained(torch.tensor(samples_glob), freeze=True)
    ne_model = torch.nn.Sequential(embd_layer, encoder)

    # Record loss values
    txtfile = open(os.path.join(args.folder_path, 'losses_pretrain.txt'), "w")
    txtfile.writelines([f'loss  attractive_loss  repulsive_loss \n'])

    # ========================= Training =========================
    for ep in range(epochs):

        embedder, z_neg_tSNE = modelUpdate(
                                    train_data=samples_glob,
                                    model=ne_model,
                                    k=args.k,
                                    lr=lr,
                                    epochs_local=1,
                                    batch_size=batch_size,
                                    n_batches=n_batches,
                                    client_graph_info=neighborhood_pretrain,
                                    client_funct_dict=None,
                                    isCent=True,
                                )
        encoder = copy.deepcopy(embedder.model[1])

        # Test on real globdata & visualize embeddings
        test_loss, test_loss_pos, test_loss_neg, z_cne_pretrain = \
            test_and_plot(args, ep, encoder, globimages, globlabels, kNN_graph_glob)
        print(f'*** [Pretrain: e{ep}]   test loss --> {test_loss} {test_loss_pos} {test_loss_neg} \n')

        # Save encoder:
        if not os.path.exists(os.path.join(args.folder_path, 'saved_encoders_pretrain')):
            os.mkdir(os.path.join(args.folder_path, 'saved_encoders_pretrain'))
        torch.save(encoder.state_dict(), os.path.join(args.folder_path, 'saved_encoders_pretrain', f'encoder_e{ep}.p'))

        # Update Encoder and Model:
        ne_model = torch.nn.Sequential(embd_layer, copy.deepcopy(encoder))

        txtfile.writelines([f'E{ep} {test_loss} {test_loss_pos} {test_loss_neg}\n'])
        txtfile.flush()

    txtfile.close()

    return copy.deepcopy(encoder).eval(), z_cne_pretrain

