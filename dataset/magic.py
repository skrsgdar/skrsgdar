import numpy as np

import logging
import os

__HEADER__ = None

__RAWDATA__ = "magic04.data"

from torch.utils.data import DataLoader
from partition import uniform_client_partition


def train_test_partition(partitions: list, train_proportion=0.8):
    train_partition, test_partition = list(), list()
    for client_partition in partitions:
        ind_split = int(len(client_partition) * train_proportion)
        train_ids, test_ids = client_partition[:ind_split], client_partition[ind_split:]
        train_partition.append(train_ids)
        test_partition.append(test_ids)
    return train_partition, test_partition


def read_file(path):
    data = list()
    n_row, n_del = 0, 0
    n_pos, n_neg = 0, 0

    mins, maxs = [None] * 10, [None] * 10

    with open(path, 'r') as file:
        for line in file:
            if line == "\n":
                continue
            if "?" in line or "none" in line:
                n_del += 1
                continue

            sample = line.replace("\n", "").split(",")

            assert len(sample) == 11, f"Length of sample is {len(sample)}, while there are 11 attributes"

            # record max and mins
            for i, value in enumerate(sample[:-1]):
                sample[i] = np.float32(value)

                if mins[i] is None or sample[i] < mins[i]:
                    mins[i] = sample[i]

                if maxs[i] is None or sample[i] > maxs[i]:
                    maxs[i] = sample[i]

            # record label
            if sample[-1] == "g":
                n_pos += 1
                sample[-1] = 1
            elif sample[-1] == "h":
                n_neg += 1
                sample[-1] = 0
            else:
                raise ValueError(f"Label {sample[-1]} is wrong")

            data.append(sample)

            n_row += 1

    # normalization
    for sample in data:
        for i, value in enumerate(sample[:-1]):
            sample[i] = (value - mins[i]) / (maxs[i] - mins[i])

    logging.info(f" {path}: Load {n_row}, remove {n_del}, total {n_row + n_del}, positive {n_pos}, negative {n_neg}")

    return data


def getMagicDataset_by_batchsz(n_client, batch_size, num_workers, drop_last=False, root="../raw_data/MAGIC/"):
    raw_data = read_file(os.path.join(root, __RAWDATA__))

    # first divided into ${n_client} clients
    partitions = uniform_client_partition(len(raw_data), n_client)
    # split into train/test for each client
    train_partition, test_partition = train_test_partition(partitions, 0.8)

    # Only one test partition for now
    test_partition = [np.concatenate(test_partition).tolist()]

    # wrap into dataset
    train_dataset = [MagicDataset(_, raw_data) for _ in train_partition]
    test_dataset = [MagicDataset(_, raw_data) for _ in test_partition]

    # wrap dataloaders
    train_loaders = [DataLoader(_,
                                shuffle=True,
                                batch_size=batch_size,
                                drop_last=drop_last,
                                num_workers=num_workers) for _ in train_dataset]
    test_loaders = [DataLoader(_,
                               shuffle=False,
                               batch_size=batch_size,
                               drop_last=drop_last,
                               num_workers=num_workers) for _ in test_dataset]
    return train_loaders, test_loaders


def getMagicDataset_by_sampling_rate(n_client, q, num_workers, drop_last=False, root="../raw_data/MAGIC/"):
    raw_data = read_file(os.path.join(root, __RAWDATA__))

    # first divided into ${n_client} clients
    partitions = uniform_client_partition(len(raw_data), n_client)
    # split into train/test for each client
    train_partition, test_partition = train_test_partition(partitions, 0.8)

    # Only one test partition for now
    test_partition = [np.concatenate(test_partition).tolist()]

    # wrap into dataset
    train_dataset = [MagicDataset(_, raw_data) for _ in train_partition]
    test_dataset = [MagicDataset(_, raw_data) for _ in test_partition]

    # wrap dataloaders
    train_loaders = [DataLoader(_,
                                shuffle=True,
                                batch_size=int(len(_) * q),
                                drop_last=drop_last,
                                num_workers=num_workers) for _ in train_dataset]
    test_loaders = [DataLoader(_,
                               shuffle=False,
                               batch_size=int(len(_) * q),
                               drop_last=drop_last,
                               num_workers=num_workers) for _ in test_dataset]

    return train_loaders, test_loaders


class MagicDataset(object):
    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __getitem__(self, index):
        sample = self.data[self.indices[index]]
        assert len(sample) == 11
        return np.array(sample[:-1]), sample[-1]

    def __len__(self):
        return len(self.indices)
