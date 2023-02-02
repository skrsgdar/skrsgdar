import logging
import os

import numpy as np

__HEADER__ = None

__RAWDATA__ = "australian.dat"

__ATTRIBUTE__ = {
    "A1": ("categorical", [0, 1]),  # 0,1
    "A2": "continuous",
    "A3": "continuous",
    "A4": ("categorical", [1, 2, 3]),  # 1-3
    "A5": ("categorical", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]),  # 1-14
    "A6": ("categorical", [1, 2, 3, 4, 5, 6, 7, 8, 9]),  # 1-9
    "A7": "continuous",
    "A8": ("categorical", [0, 1]),  # 0,1
    "A9": ("categorical", [0, 1]),  # 0,1
    "A10": "continuous",
    "A11": ("categorical", [0, 1]),  # 0,1
    "A12": ("categorical", [1, 2, 3]),  # 1-3
    "A13": "continuous",
    "A14": "continuous",
    "A15": ("target", [1, 2])  # 1,2
}

__NORMALIZE__ = [1, 2, 6, 9, 12, 13]

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

    mins, maxs = [-1] * len(__NORMALIZE__), [-1] * len(__NORMALIZE__)

    with open(path, 'r') as file:
        for line in file:
            if line == "\n":
                continue
            if "?" in line or "none" in line:
                n_del += 1
                continue

            sample = line.replace("\n", "").split(" ")

            assert len(sample) == len(
                __ATTRIBUTE__), f"Length of sample is {len(sample)}, while there are {len(__ATTRIBUTE__)} attributes"

            # record max and mins
            for i, index in enumerate(__NORMALIZE__):
                sample[index] = np.float32(sample[index])

                if mins[i] == -1 or sample[index] < mins[i]:
                    mins[i] = sample[index]

                if maxs[i] == -1 or sample[index] > maxs[i]:
                    maxs[i] = sample[index]

            # record label
            if sample[-1] == "1":
                n_pos += 1
                sample[-1] = 1
            elif sample[-1] == "0":
                n_neg += 1
                sample[-1] = 0
            else:
                raise ValueError(f"Label {sample[-1]} is wrong")

            data.append(sample)

            n_row += 1

    # normalization
    for sample in data:
        for i, index in enumerate(__NORMALIZE__):
            sample[index] = (sample[index] - mins[i]) / (maxs[i] - mins[i])

    logging.info(f" {path}: Load {n_row}, remove {n_del}, total {n_row + n_del}, positive {n_pos}, negative {n_neg}")

    return data


def getCreditDataset_by_batchsz(n_client, batch_size, num_workers, drop_last=False, root="../raw_data/CREDIT/"):
    raw_data = read_file(os.path.join(root, __RAWDATA__))

    # first divided into ${n_client} clients
    partitions = uniform_client_partition(len(raw_data), n_client)
    # split into train/test for each client
    train_partition, test_partition = train_test_partition(partitions, 0.8)

    # Only one test partition for now
    test_partition = [np.concatenate(test_partition).tolist()]

    # wrap into dataset
    train_dataset = [CreditDataset(_, raw_data) for _ in train_partition]
    test_dataset = [CreditDataset(_, raw_data) for _ in test_partition]

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


def getCreditDataset_by_sampling_rate(n_client, q, num_workers, drop_last=False, root="../raw_data/CREDIT/"):
    raw_data = read_file(os.path.join(root, __RAWDATA__))

    # first divided into ${n_client} clients
    partitions = uniform_client_partition(len(raw_data), n_client)
    # split into train/test for each client
    train_partition, test_partition = train_test_partition(partitions, 0.8)

    # Only one test partition for now
    test_partition = [np.concatenate(test_partition).tolist()]

    # wrap into dataset
    train_dataset = [CreditDataset(_, raw_data) for _ in train_partition]
    test_dataset = [CreditDataset(_, raw_data) for _ in test_partition]

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


class CreditDataset(object):
    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __getitem__(self, index):
        # obtain the sample
        sample = self.data[index]

        # check
        assert len(sample) == len(__ATTRIBUTE__)

        # one hot
        data = list()
        for value, (attr_name, attr_type) in zip(sample, __ATTRIBUTE__.items()):
            if attr_type == "continuous":
                # numeric
                data.append(np.float32(value))
            elif isinstance(attr_type, tuple):
                if attr_type[0] == "categorical":
                    # one-hot embedding
                    embedding = [np.float32(np.float32(value) == _) for _ in attr_type[1]]
                    # print(attr_type[1])
                    # print(value, type(value))
                    assert sum(embedding) == 1
                    data += embedding
                elif attr_type[0] == "target":
                    data.append(np.float32(value))
                else:
                    raise ValueError(f"{attr_type[0]} not supported")
            else:
                raise TypeError(f"{attr_type} not supported")
        data = np.array(data, dtype=np.float32)
        return data[:-1], data[-1]

    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    train_loaders, test_loaders = getCreditDataset_by_batchsz(10, 32, 3)

    print("")