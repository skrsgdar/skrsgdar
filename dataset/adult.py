import logging

from partition import uniform_client_partition
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

import torch
import os

__ATTRIBUTE__ = {
    "age": "continuous",
    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov",
                  "Without-pay", "Never-worked"],
    "fnlwgt": "continuous",
    "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                  "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"],
    "education-num": "continuous",
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                   "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                   "Priv-house-serv", "Protective-serv", "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "capital-gain": "continuous",
    "capital-loss": "continuous",
    "hours-per-week": "continuous",
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran",
                       "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal",
                       "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia",
                       "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                       "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"],
    "target": [">50K"]
}
__TESTDATA__ = "adult.test"
__TRAINDATA__ = "adult.data"
__HEADER__ = None
# The column that needs normalization
__NORMALIZE__ = [0, 2, 4, 10, 11, 12]

def read_file(path):
    """
    Load data, delete the rows with unknown value '?' and normalize data
    """
    data = list()
    n_row, n_del = 0, 0
    n_pos, n_neg = 0, 0

    with open(path, 'r') as file:
        for line in file:
            if line == '\n':
                continue
            # delete rows with unknown values
            if "?" in line:
                n_del += 1
                continue
            sample = line.replace("\n", "").split(", ")

            # numeric
            for attr_index in __NORMALIZE__:
                sample[attr_index] = np.float32(sample[attr_index])

            data.append(sample)
            n_row += 1
            if sample[-1] == ">50K":
                n_pos += 1
            elif sample[-1] == "<=50K":
                n_neg += 1
            else:
                raise ValueError(f"{sample[-1]}")

    # normalize into [0, 1]
    continues = [[sample[attr_index] for sample in data] for attr_index in __NORMALIZE__]
    mins, maxs = [min(_) for _ in continues], [max(_) for _ in continues]

    for sample in data:
        for i, attr_index in enumerate(__NORMALIZE__):
            sample[attr_index] = (sample[attr_index] - mins[i]) / (maxs[i] - mins[i])

    logging.info(f" {path}: Load {n_row}, remove {n_del}, total {n_row+n_del}, positive {n_pos}, negative {n_neg}")

    return data


def getAdultData_by_batchsz(n_client, batch_size, num_workers, drop_last=False, root="../raw_data/ADULT/"):
    # get raw data
    train_data = read_file(os.path.join(root, __TRAINDATA__))
    test_data = read_file(os.path.join(root, __TESTDATA__))

    assert len(train_data[0]) == len(__ATTRIBUTE__)

    # divided into clients
    train_partition = uniform_client_partition(len(train_data), n_client)
    test_partition = uniform_client_partition(len(test_data), n_client)

    # wrap into dataset
    train_dataset = [AdultDataset(_, train_data) for _ in train_partition]
    test_dataset = [AdultDataset(_, test_data) for _ in test_partition]

    # wrap into dataloader
    train_loaders = [DataLoader(_, batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=num_workers)
                     for _ in train_dataset]
    test_loaders = [DataLoader(_, batch_size=batch_size, shuffle=False, drop_last=drop_last, num_workers=num_workers)
                    for _ in test_dataset]

    return train_loaders, test_loaders


def getAdultData_by_sampling_rate(n_client, q, num_workers, drop_last=False, root="../raw_data/ADULT"):
    # get raw data
    train_data = read_file(os.path.join(root, __TRAINDATA__))
    test_data = read_file(os.path.join(root, __TESTDATA__))

    assert len(train_data[0]) == len(__ATTRIBUTE__)

    # divided into clients
    train_partition = uniform_client_partition(len(train_data), n_client)
    # test_partition = uniform_client_partition(len(test_data), n_client)
    test_partition = uniform_client_partition(len(test_data), 1)

    # wrap into dataset
    train_dataset = [AdultDataset(_, train_data) for _ in train_partition]
    test_dataset = [AdultDataset(_, test_data) for _ in test_partition]

    # wrap into dataloader
    train_loaders = [
        DataLoader(_, batch_size=int(len(_) * q), shuffle=True, drop_last=drop_last, num_workers=num_workers) for _ in
        train_dataset]
    test_loaders = [
        DataLoader(_, batch_size=int(len(_) * q), shuffle=False, drop_last=drop_last, num_workers=num_workers) for _ in
        test_dataset]

    return train_loaders, test_loaders


class AdultDataset(object):
    def __init__(self, indices, data):
        self.indices = indices
        self.data = data

    def __getitem__(self, index):
        # obtain the sample
        sample = self.data[self.indices[index]]

        # transform
        data = list()
        for value, (attr_name, attr_domain) in zip(sample, __ATTRIBUTE__.items()):
            if attr_domain == "continuous":
                # numeric
                data.append(np.float32(value))
            elif isinstance(attr_domain, list):
                # one-hot embedding
                embedding = [np.float32(value == _) for _ in attr_domain]
                assert sum(embedding) == 1 or attr_name == "target" and value in [">50K", "<=50K"]
                data += embedding
            else:
                raise NotImplementedError(f"Cannot found {value} in domain {attr_domain}.")
        data = np.array(data)
        return data[:-1], data[-1]

    def __len__(self):
        return len(self.indices)
