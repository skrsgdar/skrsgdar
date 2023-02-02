from dataset.adult import getAdultData_by_batchsz, getAdultData_by_sampling_rate
from dataset.bank import getBankData_by_batchsz, getBankData_by_sampling_rate
from dataset.cancer import getBreastCancer_by_batchsz, getBreastCancer_by_sampling_rate
from dataset.client import getClientDataset_by_sampling_rate, getClientDataset_by_batchsz
from dataset.credit import getCreditDataset_by_sampling_rate, getCreditDataset_by_batchsz
from dataset.magic import getMagicDataset_by_sampling_rate, getMagicDataset_by_batchsz


def get_dataloaders(data_type, n_client, num_workers, q=None, batch_size=None, drop_last=False):

    if data_type == "ADULT":
        n_in, n_out = 105, 1
        if q is not None:
            train_loaders, test_loaders = getAdultData_by_sampling_rate(n_client, q, num_workers, drop_last)
        elif batch_size is not None:
            train_loaders, test_loaders = getAdultData_by_batchsz(n_client, batch_size, num_workers, drop_last)
        else:
            raise ValueError("q and batch_size cannot be None at the same time")
    elif data_type == "BANK":
        n_in, n_out = 51, 1
        if q is not None:
            train_loaders, test_loaders = getBankData_by_sampling_rate(n_client, q, num_workers, drop_last)
        elif batch_size is not None:
            train_loaders, test_loaders = getBankData_by_batchsz(n_client, batch_size, num_workers, drop_last)
        else:
            raise ValueError("q and batch_size cannot be None at the same time")
    elif data_type == "CANCER":
        n_in, n_out = 30, 1
        if q is not None:
            train_loaders, test_loaders = getBreastCancer_by_sampling_rate(n_client, q, num_workers, drop_last)
        elif batch_size is not None:
            train_loaders, test_loaders = getBreastCancer_by_batchsz(n_client, batch_size, num_workers, drop_last)
        else:
            raise ValueError("q and batch_size cannot be None at the same time")
    elif data_type == "CREDIT":
        n_in, n_out = 43, 1
        if q is not None:
            train_loaders, test_loaders = getCreditDataset_by_sampling_rate(n_client, q, num_workers, drop_last)
        elif batch_size is not None:
            train_loaders, test_loaders = getCreditDataset_by_batchsz(n_client, batch_size, num_workers, drop_last)
        else:
            raise ValueError("q and batch_size cannot be None at the same time")
    elif data_type == "MAGIC":
        n_in, n_out = 10, 1
        if q is not None:
            train_loaders, test_loaders = getMagicDataset_by_sampling_rate(n_client, q, num_workers, drop_last)
        elif batch_size is not None:
            train_loaders, test_loaders = getMagicDataset_by_batchsz(n_client, batch_size, num_workers, drop_last)
        else:
            raise ValueError("q and batch_size cannot be None at the same time")
    elif data_type == "CLIENT":
        n_in, n_out = 93, 1
        if q is not None:
            train_loaders, test_loaders = getClientDataset_by_sampling_rate(n_client, q, num_workers, drop_last)
        elif batch_size is not None:
            train_loaders, test_loaders = getClientDataset_by_batchsz(n_client, batch_size, num_workers, drop_last)
        else:
            raise ValueError("q and batch_size cannot be None at the same time")
    else:
        raise NotImplementedError()

    return train_loaders, test_loaders, n_in, n_out
