
import numpy as np

def uniform_client_partition(n_samples, n_client):
    """Uniformly divide data into ${n_client} clients

    :param n_samples:
    :param n_client:
    :return:
    """
    inds = np.arange(n_samples)
    np.random.shuffle(inds)
    partitions = np.array_split(inds, n_client)
    return partitions
