import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def metric_continuity(D_high, D_low, k):
    # print(D_high.shape, D_low.shape)
    assert D_high.shape == D_low.shape, 'D_high != D_low shape!'
    N_SAMPLES = D_high.shape[0]

    n = N_SAMPLES

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(N_SAMPLES):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2/(n*k*(2*n - 3*k - 1))*sum_i)).squeeze())



def metric_trustworthiness(D_high, D_low, k):
    assert D_high.shape == D_low.shape, 'D_high != D_low shape!'
    N_SAMPLES = D_high.shape[0]

    n = N_SAMPLES

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    return float((1 - (2/(n*k*(2*n - 3*k - 1))*sum_i)).squeeze())



def metric_neighborhood_hit(X_low, y, k):
    print(y.shape)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_low, y)

    neighbors = knn.kneighbors(X_low, return_distance=False)
    return np.mean(np.mean((y[neighbors] == np.tile(y.reshape((-1, 1)), k)).astype('uint8'), axis=1))



def compare_highD_lowD_dists(D_high, D_low):
    return np.corrcoef(D_high.flatten(), D_low.flatten())[0,1]


