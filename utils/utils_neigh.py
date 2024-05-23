from scipy import sparse
import numpy as np
from scipy.sparse import lil_matrix
from annoy import AnnoyIndex
import scipy.spatial.distance as distance
from sklearn.neighbors import NearestNeighbors



def build_kNN_graph(dataset, n_nbrs):
    # shapes = dataset.shape
    dataset = dataset.reshape(dataset.shape[0], -1)
    shapes = dataset.shape
    
    # create approximate NN search tree
    print(f"Computing approximate k-NN graph:   n_nbrs={n_nbrs}")
    annoy = AnnoyIndex(shapes[1], metric="euclidean")
    for i, x in enumerate(dataset): 
        annoy.add_item(i, x)

    annoy.build(50)

    # construct the adjacency matrix for the graph
    min_dist_set = []
    adj = lil_matrix((shapes[0], shapes[0]))
    for i in range(shapes[0]):
        neighs_, dists = annoy.get_nns_by_item(i, n_nbrs + 1, include_distances=True)
        neighs = neighs_[1:]

        # min_dist_set.append(np.mean(dists[1:]))
        min_dist_set.append(np.min(dists[1:]))

        adj[i, neighs] = 1
        adj[neighs, i] = 1  # symmetrize on the fly

    return annoy, adj, np.array(min_dist_set)



def neighborhood_cent(kNN_graph):
    pos_inds_dict = {'self': []}
    neg_inds_dict = {'self': []}
    neigh_counts_dict = {'self': []}

    graph_glob_arr = kNN_graph.toarray()
    N = graph_glob_arr.shape[0]

    for j in range(N):
        neighs = list(np.where(graph_glob_arr[j, :] == 1)[0])
        pos_inds_dict['self'].append(neighs)
        neg_inds_dict['self'] = [*range(N)]
        neigh_counts_dict['self'].append(len(neighs))

    return pos_inds_dict, neg_inds_dict, neigh_counts_dict



def neighborhood_clients(client_data_list, local_id, local_images, k):
    n_clients = len(client_data_list)
    n_local = len(local_images)

    pos_inds_dict = {'self': []}
    neg_inds_dict = {'self': []}
    neigh_counts_dict = {'self': []}


    globimages_ordered = [local_images]  # make current client data at the fist place
    for j in range(n_clients):
        if j != local_id:
            globimages_ordered += [client_data_list[j]]

    _, graph_glob, mean_dist_arr = build_kNN_graph(dataset=np.vstack(globimages_ordered), n_nbrs=k)
    graph_glob_arr = graph_glob.toarray()

    # Find neighs within local own data kNN (globally normalized)
    neigh_mat = graph_glob_arr[0:n_local, 0:n_local]
    for j in range(n_local):
        neighs = list(np.where(neigh_mat[j, :] == 1)[0])
        pos_inds_dict['self'].append(neighs)

    neg_inds_dict['self'] = [*range(n_local)]

    # Find neighs of X_local within global data
    neigh_mat_2all = graph_glob_arr[0:n_local, :]
    neigh_counts_dict['self'] = np.sum(neigh_mat_2all, axis=1).astype(np.int)

    return pos_inds_dict, neg_inds_dict, neigh_counts_dict, mean_dist_arr



def get_max_dists(dataset, n_nbrs):
    shapes = dataset.shape
    # create approximate NN search tree
    print(f"Computing approximate k-NN graph:   n_nbrs={n_nbrs}")
    annoy = AnnoyIndex(shapes[1], metric="euclidean")
    for i, x in enumerate(dataset):
        annoy.add_item(i, x)

    annoy.build(50)

    # construct the adjacency matrix for the graph
    max_dist_set = []
    adj = lil_matrix((shapes[0], shapes[0]))
    for i in range(shapes[0]):
        neighs_, dists = annoy.get_nns_by_item(i, n_nbrs + 1, include_distances=True)
        neighs = neighs_[1:]
        max_dist_set.append(np.max(dists[1:]))

        adj[i, neighs] = 1
        adj[neighs, i] = 1  # symmetrize on the fly

    return adj, np.array(max_dist_set)




def local_knn_radius(local_images, k):
    n_local = len(local_images)

    pos_inds_dict = {'self': []}
    neg_inds_dict = {'self': []}
    neigh_counts_dict = {'self': []}

    graph_local, max_dist_set = get_max_dists(dataset=np.vstack(local_images), n_nbrs=k)

    pos_inds_dict['self'] = graph_local.rows
    neg_inds_dict['self'] = [*range(n_local)]

    # Find neighs of X_local within global data
    neigh_counts_dict['self'] = np.array([len(lst) for lst in graph_local.rows])

    return max_dist_set




def neighborhood_clients_2(local_images, k):
    n_local = len(local_images)

    pos_inds_dict = {'self': []}
    neg_inds_dict = {'self': []}
    neigh_counts_dict = {'self': []}


    _, graph_local, mean_dist_arr = build_kNN_graph(dataset=np.vstack(local_images), n_nbrs=k)

    pos_inds_dict['self'] = graph_local.rows

    neg_inds_dict['self'] = [*range(n_local)]

    # Find neighs of X_local within global data
    neigh_counts_dict['self'] = np.array([len(lst) for lst in graph_local.rows])

    return pos_inds_dict, neg_inds_dict, neigh_counts_dict, mean_dist_arr




def find_1NN(X, Y, type=None):
    if type == 'loo':
        # Fit the model using data from Y
        nbrs = NearestNeighbors(n_neighbors=2).fit(Y)
        distances, indices = nbrs.kneighbors(X)
        return np.array([[index[1]] for index in indices])
    else:
        # Fit the model using data from Y
        nbrs = NearestNeighbors(n_neighbors=1).fit(Y)
        distances, indices = nbrs.kneighbors(X)
        return np.array([[index[0]] for index in indices])  #[Y[index[0]] for index in indices]




def find_kNN(X, Y, k, type=None):
    if type == 'loo':
        # Fit the model using data from Y
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(Y)
        distances, indices = nbrs.kneighbors(X)
        return np.array([index[1:] for index in indices])
    else:
        # Fit the model using data from Y
        nbrs = NearestNeighbors(n_neighbors=k).fit(Y)
        distances, indices = nbrs.kneighbors(X)
        return indices





def find_1NN_wRadius(X, Y, max_dist_set, type=None):
    indices_1NN = []
    if type == 'loo':
        # Fit the model using data from Y
        nbrs = NearestNeighbors(n_neighbors=2).fit(Y)
        distances, indices = nbrs.kneighbors(X)
        for i, (ind, dist) in enumerate(zip(indices, distances)):
            if dist[1] < max_dist_set[ind[1]]:
                indices_1NN.append([ind[1]])
            else:
                indices_1NN.append([])
        return np.array(indices_1NN)
    else:
        # Fit the model using data from Y
        nbrs = NearestNeighbors(n_neighbors=1).fit(Y)
        distances, indices = nbrs.kneighbors(X)
        for i, (ind, dist) in enumerate(zip(indices, distances)):
            if dist[0] < max_dist_set[ind[0]]:
                indices_1NN.append([ind[0]])
            else:
                indices_1NN.append([])
        return np.array(indices_1NN)



# def find_1NN_dist_wRadius(X, Y, max_dist_set, type=None):
#     dists_1NN = []
#     if type == 'loo':
#         # Fit the model using data from Y
#         nbrs = NearestNeighbors(n_neighbors=2).fit(Y)
#         distances, indices = nbrs.kneighbors(X)
#         for i, (ind, dist) in enumerate(zip(indices, distances)):
#             if dist[1] < max_dist_set[ind[1]]:
#                 dists_1NN.append([dist[1]])
#             else:
#                 dists_1NN.append([0.0])
#         return np.array(dists_1NN)
#     else:
#         # Fit the model using data from Y
#         nbrs = NearestNeighbors(n_neighbors=1).fit(Y)
#         distances, indices = nbrs.kneighbors(X)
#         for i, (ind, dist) in enumerate(zip(indices, distances)):
#             if dist[0] < max_dist_set[ind[0]]:
#                 dists_1NN.append([dist[0]])
#             else:
#                 dists_1NN.append([0.0])
#         return np.array(dists_1NN)



def find_1NN_dist(X, Y, type=None):
    if type == 'loo':
        # Fit the model using data from Y
        nbrs = NearestNeighbors(n_neighbors=2).fit(Y)
        distances, indices = nbrs.kneighbors(X)
        return np.array([[dist[1]] for dist in distances])
    else:
        # Fit the model using data from Y
        nbrs = NearestNeighbors(n_neighbors=1).fit(Y)
        distances, indices = nbrs.kneighbors(X)
        return np.array([[dist[0]] for dist in distances])



def get_neighs_of_queryNN(local_images, dist_set, X_query_NN):
    n_local = len(local_images)
    n_query = len(X_query_NN)
    NN_inds_in_client = [[] for _ in range(n_query)]

    for i, data in enumerate(local_images):
        thred_dist = dist_set[i]
        dist_row = distance.cdist(local_images[i:i + 1], X_query_NN, 'euclidean')[0]
        inds_in_query = np.where(0 < dist_row < thred_dist)[0]
        for ind in inds_in_query:
            NN_inds_in_client[ind].append(i)

    return np.array(NN_inds_in_client)




def neighborhood_query_2d(Z_query, Z_local, k):

    n_query = len(Z_query)
    n_local = len(Z_local)
    Z_combo = np.vstack([Z_query, Z_local])

    pos_inds_dict = {'query': []}
    neg_inds_dict = {'query': []}
    neigh_counts_dict = {'query': []}

    _, graph_combo, __ = build_kNN_graph(dataset=Z_combo, n_nbrs=k)
    graph_combo_arr = graph_combo.toarray()

    # Find neighs of Z_query within Z_local / Z_combo
    neigh_mat_query2local = graph_combo_arr[0:n_query, n_query:]

    pos_inds_dict['query'] = [list(np.where(neigh_mat_query2local[j,:] == 1)[0]) for j in range(n_query)]

    # Find negative samples within Z_local
    neg_inds_dict['query'] = [*range(n_local)]

    # Find neighs of X_local within local & query
    neigh_mat_query2combo = graph_combo_arr[0:n_query, :]
    neigh_counts_dict['query'] = np.sum(neigh_mat_query2combo, axis=1).astype(np.int)

    return pos_inds_dict, neg_inds_dict, neigh_counts_dict




def neighborhood_query_2d_v2(Z_query, Z_local, k, n_nbrs):

    n_query = len(Z_query)
    n_local = len(Z_local)
    Z_combo = np.vstack([Z_query, Z_local])

    pos_inds_dict = {'query': []}
    neg_inds_dict = {'query': []}
    neigh_counts_dict = {'query': []}

    _, graph_combo, __ = build_kNN_graph(dataset=Z_combo, n_nbrs=n_nbrs)
    graph_combo_arr = graph_combo.toarray()

    # Find neighs of Z_query within Z_local / Z_combo
    neigh_mat_query2local = graph_combo_arr[0:n_query, n_query:]

    pos_inds_dict['query'] = [list(np.where(neigh_mat_query2local[j,:] == 1)[0]) for j in range(n_query)]

    # Find negative samples within Z_local
    neg_inds_dict['query'] = [*range(n_local)]

    # Find neighs of X_local within local & query
    # neigh_mat_query2combo = graph_combo_arr[0:n_query, :]
    neigh_counts_dict['query'] = np.array([k] * n_query, dtype=int)

    return pos_inds_dict, neg_inds_dict, neigh_counts_dict




# def neighborhood_query_2d_v2(Z_query, Z_local, k):
#
#     n_query = len(Z_query)
#     n_local = len(Z_local)
#
#     pos_inds_dict = {'query': []}
#     neg_inds_dict = {'query': []}
#     neigh_counts_dict = {'query': []}
#
#     adj_local2query = np.zeros((n_local, n_query), dtype=np.int)
#
#     for i in range(n_local):
#         dist_row = distance.cdist(Z_local[i:i + 1], Z_query, 'euclidean')[0]
#         inds_topK = np.argsort(dist_row)[:k]
#         adj_local2query[i, inds_topK] = 1
#
#     for j in range(n_query):
#         neighs = list(np.where(adj_local2query[:,j] == 1)[0])
#         pos_inds_dict['query'].append(neighs)
#
#     # Find negative samples within Z_local
#     neg_inds_dict['query'] = [*range(n_local)]
#
#     # Find num of all neighs of z_q within Z_query
#     _, graph_query, __ = build_kNN_graph(dataset=Z_query, n_nbrs=k)
#     neigh_counts_dict['query'] = np.array([sum(row) for row in graph_query.data]).astype(np.int)
#
#     return pos_inds_dict, neg_inds_dict, neigh_counts_dict
#



def neighborhood_query_2d_v3(query_neigh_dict, Z_query, Z_local, k):

    n_query = len(Z_query)
    n_local = len(Z_local)

    pos_inds_dict = {'query': []}
    neg_inds_dict = {'query': []}
    neigh_counts_dict = {'query': []}

    pos_inds_dict['query'] = [query_neigh_dict[key] for key in query_neigh_dict.keys()]

    # Find negative samples within Z_local
    neg_inds_dict['query'] = [*range(n_local)]

    neigh_counts_dict['query'] = np.array([k+5] * n_query)

    return pos_inds_dict, neg_inds_dict, neigh_counts_dict




def neighborhood_fakeDecent(local_data, local_id, client_data_list, n_clients, n_nbrs):
    # make current client data at the fist place:
    data_glob = [local_data]
    N_local = len(local_data)
    client_sizes = [N_local]
    client_start_inds = [0]
    client_end_inds = [N_local]
    client_inds_new = [local_id]

    for j in range(n_clients):
        if j != local_id:
            data_glob += [client_data_list[j]]
            client_sizes.append(len(client_data_list[j]))
            client_start_inds.append(client_end_inds[-1])
            client_end_inds.append(client_end_inds[-1] + len(client_data_list[j]))
            client_inds_new.append(j)

    _, graph_glob_temp, __ = build_kNN_graph(dataset=np.vstack(data_glob), n_nbrs=n_nbrs)
    graph_glob_arr = graph_glob_temp.toarray()

    graph_set = {}
    summ = np.sum(graph_glob_arr[:N_local,:])
    test_sum_set = []

    neg_inds_dict = {}
    pos_inds_dict = {}

    scale_mat_pos = {}

    for i, cid in enumerate(client_inds_new):
        scale_mat_pos[cid] = []

        end_id = client_end_inds[i]
        start_id = client_start_inds[i]

        matrix_curr = graph_glob_arr[:N_local, start_id:end_id]
        graph_kNN_curr = sparse.lil_matrix(matrix_curr)
        graph_set[cid] = graph_kNN_curr

        n_total_pos = np.sum(graph_glob_arr[:N_local, :], axis=1)

        pos_inds_dict[cid] = []
        for j in range(matrix_curr.shape[0]):
            nbr_inds = list(np.where(matrix_curr[j,:] == 1)[0])
            scale_mat_pos[cid].append(n_total_pos[j])       # total number of positive neighbors

            nbr_inds = [ind + start_id for ind in nbr_inds]      # ind + start_id
            pos_inds_dict[cid].append(nbr_inds)
        pos_inds_dict[cid] = np.array(pos_inds_dict[cid], dtype=object)

        test_sum = np.sum(matrix_curr)
        test_sum_set.append(test_sum)

        neg_inds = [*range(start_id, end_id)]
        neg_inds_dict[cid] = neg_inds

    assert summ == sum(test_sum_set)

    # # Check all client data equal:
    # for size1 in client_sizes:
    #     for size2 in client_sizes:
    #         assert size1 == size2

    return np.vstack(data_glob), neg_inds_dict, pos_inds_dict, scale_mat_pos




def highD_NN_query(local_images, Z_local, X_query_NN, Z_query_NN, k, mean_dist_arrs):

    n_query = len(Z_query_NN)
    n_local = len(Z_local)
    X_combo = np.vstack([X_query_NN, local_images])

    pos_inds_dict = {'query': []}
    neg_inds_dict = {'query': []}
    neigh_counts_dict = {'query': []}

    _, graph_combo, __ = build_kNN_graph(dataset=X_combo, n_nbrs=int(n_nbrs))
    graph_combo_arr = graph_combo.toarray()

    # Find neighs of Z_query within Z_local / Z_combo
    neigh_mat_query2local = graph_combo_arr[0:n_query, n_query:]
    pos_inds_dict['query'] = [list(np.where(neigh_mat_query2local[j,:] == 1)[0]) for j in range(n_query)]

    # Find negative samples within Z_local
    neg_inds_dict['query'] = [*range(n_local)]

    # Find neighs of X_local within local & query
    neigh_counts_dict['query'] = np.array([k] * n_query, dtype=int)

    # neigh_mat_query2combo = graph_combo_arr[0:n_query, :]
    # neigh_counts_dict['query'] = np.sum(neigh_mat_query2combo, axis=1).astype(np.int)

    return pos_inds_dict, neg_inds_dict, neigh_counts_dict



