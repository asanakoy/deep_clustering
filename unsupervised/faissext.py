import numpy as np
import faiss
import sys
import time
import warnings

if not sys.warnoptions:
    # suppress pesky PIL EXIF warnings
    warnings.simplefilter("once")
    warnings.filterwarnings("ignore", message="(Possibly )?corrupt EXIF data.*")
    warnings.filterwarnings("ignore", message="numpy.dtype size changed.*")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed.*")


def preprocess_features(x, d=256):
    """
    Calculate PCA + Whitening + L2 normalization for each vector

    Args:
        x (ndarray): N x D, where N is number of vectors, D - dimensionality
        d (int): number of output dimensions (how many principal components to use).
    Returns:
        transformed [N x d] matrix xt .
    """
    n, orig_d = x.shape
    pcaw = faiss.PCAMatrix(d_in=orig_d, d_out=d, eigen_power=-0.5, random_rotation=False)
    pcaw.train(x)
    assert pcaw.is_trained
    print 'Performing PCA + whitening'
    x = pcaw.apply_py(x)
    print 'x.shape after PCA + whitening:', x.shape
    l2normalization = faiss.NormalizationTransform(d, 2.0)
    print 'Performing L2 normalization'
    x = l2normalization.apply_py(x)
    return x


def train_kmeans(x, num_clusters=1000, num_gpus=1):
    """
    Runs k-means clustering on one or several GPUs
    """
    d = x.shape[1]
    kmeans = faiss.Clustering(d, num_clusters)
    kmeans.verbose = True
    kmeans.niter = 20

    # otherwise the kmeans implementation sub-samples the training set
    kmeans.max_points_per_centroid = 10000000

    res = [faiss.StandardGpuResources() for i in range(num_gpus)]

    flat_config = []
    for i in range(num_gpus):
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = i
        flat_config.append(cfg)

    if num_gpus == 1:
        index = faiss.GpuIndexFlatL2(res[0], d, flat_config[0])
    else:
        indexes = [faiss.GpuIndexFlatL2(res[i], d, flat_config[i])
                   for i in range(num_gpus)]
        index = faiss.IndexProxy()
        for sub_index in indexes:
            index.addIndex(sub_index)

    # perform the training
    kmeans.train(x, index)
    print 'Total number of indexed vectors (after kmeans.train()):', index.ntotal
    centroids = faiss.vector_float_to_array(kmeans.centroids)

    objective = faiss.vector_float_to_array(kmeans.obj)
    print 'Objective values per iter:', objective
    print "Final objective: %.4g" % objective[-1]

    # TODO: return cluster assignment

    return centroids.reshape(num_clusters, d)


def compute_cluster_assignment(centroids, x):
    assert centroids is not None, "should train before assigning"
    d = centroids.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(centroids)
    distances, labels = index.search(x, 1)
    return labels.ravel()


def do_clustering(features, num_clusters, num_gpus=None):
    if num_gpus is None:
        num_gpus = faiss.get_num_gpus()
    print 'FAISS: using {} GPUs'.format(num_gpus)
    features = np.asarray(features.reshape(features.shape[0], -1), dtype=np.float32)
    features = preprocess_features(features)

    print 'Run FAISS clustering...'
    t0 = time.time()
    centroids = train_kmeans(features, num_clusters, num_gpus)
    print 'Compute cluster assignment'
    labels = compute_cluster_assignment(centroids, features)
    print 'centroids.shape:', centroids.shape
    print 'labels.shape:', labels.shape
    t1 = time.time()
    print "Total elapsed time: %.3f m" % ((t1 - t0) / 60.0)
    return labels


def example():
    k = 1000
    ngpu = 1

    x = np.random.rand(1000000, 512)
    print "reshape"
    x = x.reshape(x.shape[0], -1).astype('float32')
    x = preprocess_features(x)

    print "run"
    t0 = time.time()
    centroids = train_kmeans(x, k, ngpu)
    print 'compute_cluster_assignment'
    labels = compute_cluster_assignment(centroids, x)
    print 'centroids.shape:', centroids.shape
    print 'labels.shape:', labels.shape
    t1 = time.time()

    print "total runtime: %.3f s" % (t1 - t0)


if __name__ == '__main__':
    example()
