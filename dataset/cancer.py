import logging
import os
import numpy as np
from partition import uniform_client_partition
from torch.utils.data import DataLoader

__RAWDATA__ = "wdbc.data"

__HEADER__ = None


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

    mins, maxs = [-1] * 30, [-1] * 30

    with open(path, 'r') as file:
        for line in file:
            if line == "\n":
                continue
            # delete rows with unknown values
            if "?" in line or "none" in line:
                n_del += 1
                continue
            # the first column is ID
            sample = line.replace("\n", "").split(",")[1:]
            label = sample[0]
            attributes = sample[1:]

            # target
            if label == "M":
                label = 1
                n_pos += 1
            elif label == "B":
                label = 0
                n_neg += 1
            else:
                raise ValueError(f"{label} not supported")

            # attributes
            for i, value in enumerate(attributes):
                attributes[i] = np.float32(value)

                if mins[i] == -1 or attributes[i] < mins[i]:
                    mins[i] = attributes[i]
                if maxs[i] == -1 or attributes[i] > maxs[i]:
                    maxs[i] = attributes[i]

            data.append(attributes + [label])

            n_row += 1

    # normalize all values
    for sample in data:
        for i, value in enumerate(sample[:-1]):
            sample[i] = (value - mins[i]) / (maxs[i] - mins[i])

    logging.info(f" {path}: Load {n_row}, remove {n_del}, total {n_row + n_del}, positive {n_pos}, negative {n_neg}")

    return data


def getBreastCancer_by_batchsz(n_client, batch_size, num_workers, drop_last=False, root="../raw_data/CANCER/"):
    # get raw data
    raw_data = read_file(os.path.join(root, __RAWDATA__))

    # first divided into ${n_client} clients
    partitions = uniform_client_partition(len(raw_data), n_client)
    # split into train/test for each client
    train_partition, test_partition = train_test_partition(partitions, 0.8)

    # Only one test partition for now
    test_partition = [np.concatenate(test_partition).tolist()]

    # wrap datasets
    train_dataset = [CancerDataset(_, raw_data) for _ in train_partition]
    test_dataset = [CancerDataset(_, raw_data) for _ in test_partition]

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


def getBreastCancer_by_sampling_rate(n_client, q, num_workers, drop_last=False, root="../raw_data/CANCER/"):
    # get raw data
    raw_data = read_file(os.path.join(root, __RAWDATA__))

    # first divided into ${n_client} clients
    partitions = uniform_client_partition(len(raw_data), n_client)
    # split into train/test for each client
    train_partition, test_partition = train_test_partition(partitions, 0.8)

    # Only one test partition for now
    test_partition = [np.concatenate(test_partition).tolist()]

    # wrap datasets
    train_dataset = [CancerDataset(_, raw_data) for _ in train_partition]
    test_dataset = [CancerDataset(_, raw_data) for _ in test_partition]

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


class CancerDataset(object):
    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __getitem__(self, index):
        # obtain the sample
        sample = self.data[self.indices[index]]
        return np.array(sample[:-1]), sample[-1]

    def __len__(self):
        return len(self.indices)


if __name__ == '__main__':
    train_loaders, test_loaders = getBreastCancer_by_batchsz(10, 32, 3)

    print("")
