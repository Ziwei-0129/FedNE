import numpy as np
import random
from utils.utils_neigh import build_kNN_graph
from sklearn.mixture import GaussianMixture




def local_augmentation(local_images, k, n_new):

    _, graph, __ = build_kNN_graph(dataset=local_images, n_nbrs=k)
    local_neighs = graph.rows

    new_samples = []
    for i, image in enumerate(local_images):
        nn_ids = np.array(local_neighs[i])

        if len(nn_ids) > 0:
            nn_images = local_images[nn_ids]
            nn_images = np.vstack([image, nn_images])

            num_nns = max(len(nn_ids) // 2, 2)
            for j in range(n_new):
                points = nn_images[np.random.choice(len(nn_ids), num_nns, replace=False)]
                new_point = np.mean(points, axis=0)
                # noise = np.random.normal(0, 0.1, new_point.shape)
                # new_point = new_point + noise
                new_samples.append(new_point)

    return np.array(new_samples)




def gmm_sample(gm, n_sample):
    x, y = gm.sample(n_sample)
    return x


def extract_gmm(data, n_components=2, covariance_type='full'):
    gm = GaussianMixture(n_components=n_components, covariance_type=covariance_type).fit(data)
    return gm


def gmm_sample_mnist(gmm_list, n_samples):
    new_samples = []
    for _gm in gmm_list:
        X = gmm_sample(_gm, n_sample=n_samples)
        new_image = np.clip(X, 0, 1)
        new_samples.append(new_image.astype(np.float32))
    return np.vstack(new_samples)



def generate_gmm_samples(glob_data, n_components, n_samples):
    gm_glob = extract_gmm(glob_data, n_components=n_components, covariance_type='full')
    samples = gmm_sample_mnist([gm_glob], n_samples)
    return samples



def generate_gmm_samples_2d(glob_data, n_components, n_samples):
    gm_glob = extract_gmm(glob_data, n_components=n_components, covariance_type='full')
    X = gmm_sample(gm_glob, n_sample=n_samples)
    samples = X.astype(np.float32)
    return samples



def generate_gmm_samples_2d_v2(client_labels, embedding_set, client_size_list):
    samples = []
    # 一个class 一个GMM
    for cid, (labels, embeddings) in enumerate(zip(client_labels, embedding_set)):
        n_components = 0
        # estimate total gmms
        for num in np.unique(labels):
            freq = np.where(labels == num)[0]
            if 30 < len(freq) < 500:
                n_components += 1
            elif len(freq) >= 500:
                n_components += 2
        gm = extract_gmm(embeddings, n_components=n_components, covariance_type='full')
        # sample data from gmms
        z = gmm_sample(gm, n_sample=client_size_list[cid])
        z = z.astype(np.float32)
        samples.append(z)
    samples = np.vstack(samples)
    return samples




def get_globdata_forPretrain(mean_clients, cov_clients, n_samples):
    new_samples = []
    for cid in mean_clients.keys():
        mean_, cov_ = mean_clients[cid].squeeze(), cov_clients[cid].squeeze()
        X = np.random.multivariate_normal(mean_, cov_, size=n_samples)
        new_image = np.clip(X, 0, 1)
        new_samples.append(new_image.astype(np.float32))
    return np.vstack(new_samples)



def sample_neighborhood_2D(center, range_x, range_y, step_size):

    # Determine start and end for each axis
    x_start = center[0] - range_x
    x_end = center[0] + range_x + step_size  # Add step_size to include the end value
    y_start = center[1] - range_y
    y_end = center[1] + range_y + step_size  # Add step_size to include the end value

    # Generate 1D coordinates
    x_coords = np.arange(x_start, x_end, step_size)
    y_coords = np.arange(y_start, y_end, step_size)

    # Generate 2D grid
    xx, yy = np.meshgrid(x_coords, y_coords)

    # Stack to get a single 2D numpy array
    grid = np.stack((xx.ravel(), yy.ravel()), axis=-1)

    return grid



