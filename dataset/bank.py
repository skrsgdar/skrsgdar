import pandas as pd
import numpy as np
import os

from partition import uniform_client_partition
from torch.utils.data import DataLoader

import logging

__ATTRIBUTE__ = {
    "age": "numeric",
    "job": ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student", "blue-collar",
            "self-employed", "retired", "technician", "services"],
    "marital": ["married", "divorced", "single"],
    "education": ["unknown", "secondary", "primary", "tertiary"],
    "default": ["yes", "no"],
    "balance": "numeric",
    "housing": ["yes", "no"],
    "loan": ["yes", "no"],
    "contact": ["unknown", "telephone", "cellular"],
    "day": "numeric",
    "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "duration": "numeric",
    "campaign": "numeric",
    "pdays": "numeric",
    "previous": "numeric",
    "poutcome": ["unknown", "other", "failure", "success"],
    "y": ["yes"]
}

__RAWDATA__ = "bank/bank-full.csv"
__HEADER__ = 0
__NORMALIZE__ = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

def train_test_partition(partitions: list, train_proportion=0.8):
    train_partition, test_partition = list(), list()
    for client_partition in partitions:
        ind_split = int(len(client_partition) * train_proportion)
        train_ids, test_ids = client_partition[:ind_split], client_partition[ind_split:]
        train_partition.append(train_ids)
        test_partition.append(test_ids)
    return train_partition, test_partition


def getBankData_by_batchsz(n_client, batch_size, num_workers, drop_last=False, root="../raw_data/BANK/"):
    raw_data = pd.read_csv(os.path.join(root, __RAWDATA__),
                           sep=';',
                           engine="python",
                           header=__HEADER__)
    distribution = raw_data.y.value_counts()
    logging.info(f"Data: total {len(raw_data)}, positive {distribution['yes']}, negative {distribution['no']}")

    # normalize here
    for attr in __NORMALIZE__:
        min_v = float(raw_data[attr].min(axis=0))
        max_v = float(raw_data[attr].max(axis=0))
        raw_data[attr] = (raw_data[attr] - min_v) / (max_v - min_v)

    # first divided into ${n_client} clients
    partitions = uniform_client_partition(len(raw_data), n_client)
    # split into train/test for each client
    train_partition, test_partition = train_test_partition(partitions, 0.8)

    # wrap datasets
    train_dataset = [BankDataset(_, raw_data) for _ in train_partition]
    test_dataset = [BankDataset(_, raw_data) for _ in test_partition]
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


def getBankData_by_sampling_rate(n_client, q, num_workers, drop_last=False, root="../raw_data/BANK/"):
    raw_data = pd.read_csv(os.path.join(root, __RAWDATA__),
                           sep=';',
                           engine="python",
                           header=__HEADER__)
    distribution = raw_data.y.value_counts()
    logging.info(f"Data: total {len(raw_data)}, positive {distribution['yes']}, negative {distribution['no']}")

    # normalize here
    for attr in __NORMALIZE__:
        min_v = float(raw_data[attr].min(axis=0))
        max_v = float(raw_data[attr].max(axis=0))
        raw_data[attr] = (raw_data[attr] - min_v) / (max_v - min_v)

    # first divided into ${n_client} clients
    partitions = uniform_client_partition(len(raw_data), n_client)
    # split into train/test for each client
    train_partition, test_partition = train_test_partition(partitions, 0.8)

    # Only one test partition for now
    test_partition = [np.concatenate(test_partition).tolist()]

    # wrap datasets
    train_dataset = [BankDataset(_, raw_data) for _ in train_partition]
    test_dataset = [BankDataset(_, raw_data) for _ in test_partition]
    # wrap dataloaders
    train_loaders = [DataLoader(_,
                                shuffle=True,
                                batch_size=int(len(_)*q),
                                drop_last=drop_last,
                                num_workers=num_workers) for _ in train_dataset]
    test_loaders = [DataLoader(_,
                               shuffle=False,
                               batch_size=int(len(_)*q),
                               drop_last=drop_last,
                               num_workers=num_workers) for _ in test_dataset]

    return train_loaders, test_loaders


class BankDataset(object):
    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __getitem__(self, index):
        # obtain the sample
        sample = self.data.iloc[self.indices[index]]

        # transform
        data = list()
        for value, (attr_name, attr_domain) in zip(sample, __ATTRIBUTE__.items()):
            if attr_domain == "numeric":
                # numeric
                data.append(np.float32(value))
            elif isinstance(attr_domain, list):
                # one-hot embedding
                embedding = [np.float32(value == _) for _ in attr_domain]
                assert sum(embedding) == 1 or attr_name == "y" and value in ["yes", "no"]
                data += embedding
            else:
                raise NotImplementedError(f"Cannot found {value} in domain {attr_domain}.")
        data = np.array(data)
        return data[:-1], data[-1]

    def __len__(self):
        return len(self.indices)
