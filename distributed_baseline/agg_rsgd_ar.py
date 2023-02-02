import sys

sys.path.append('..')

from model import LR
from copy import deepcopy
import random
import numpy as np
import torch
from utils import discrete_tensor
from utils import skellam_noise, str2bool
from utils import criterion
from builder import get_dataloaders

import argparse
import json
import logging

logging.getLogger().setLevel(logging.INFO)


def local_train(model,
                dataloader,
                local_update_epochs,
                local_mini_batches,
                optimizer,
                lr,
                device,
                weight_decay,
                scale_noise,
                tau):
    model.train()
    # init optimizer
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_samples, loss_total = 0., 0.

    h = 0.
    n_avg = 0.
    avg_params = dict()
    for id_epoch in range(1, local_update_epochs+1):
        h += 1.
        # lr decay
        for g in opt.param_groups:
            g['lr'] = args.lr / h

        n_mini_batch = 0
        while n_mini_batch < local_mini_batches:
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                # forward
                pred = model(x)
                if len(y.size()) == 1:
                    y = torch.unsqueeze(y, dim=-1)
                loss = criterion(pred, y)

                # backward
                opt.zero_grad()
                loss.backward()
                opt.step()

                n_samples += len(x)
                loss_total += loss.item() * len(x)

                # record the averaged params
                with torch.no_grad():
                    n_avg += 1.
                    for name, param in model.named_parameters():
                        avg_params[name] = avg_params.get(name, 0.) + (param.data.detach()-avg_params.get(name, 0.)) / n_avg

                n_mini_batch += 1
                if n_mini_batch >= local_mini_batches:
                    break

        # load average parameters for the last args.tau * len(train_loader)
        if id_epoch % tau == 0:
            model.load_state_dict(avg_params)
            n_avg = 0.
            h = 0.
            avg_params = dict()

    with torch.no_grad():
        for param in model.parameters():
            param.data += torch.normal(mean=0., std=scale_noise, size=param.size(), device=param.device)

    return model.state_dict(), n_samples, loss_total / n_samples


def local_test(model,
               dataloader,
               device):
    model.eval()

    with torch.no_grad():
        loss_total = 0.
        n_correct = 0.

        for id_batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            # forward
            pred = model(x)
            if len(y.size()) == 1:
                y = torch.unsqueeze(y, dim=-1)
            loss = criterion(pred, y)

            predict = 1. / (1. + torch.exp(- pred))

            n_correct += torch.sum(torch.sign(predict - 0.5) * (y - 0.5) * 2 == 1).item()
            loss_total += loss.item() * len(y)

    return loss_total, n_correct, len(dataloader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", help="which device to run on, choice from ['CPU', '${INDEX_OF_GPU}']", type=str)
    parser.add_argument("--seed", help="random seed", type=int, default=1234)

    # DATA
    parser.add_argument("--batch_size", help="batch size of the data", type=int)
    parser.add_argument("--data_type", help="which data to use", type=str,
                        choices=["ADULT", "BANK", "IPUMS-US", "IPUMS-BR", "CANCER", "CREDIT", "MAGIC",  "CLIENT"])
    parser.add_argument("--num_workers", type=int, default=8)

    # FEDERATED TRAINING
    parser.add_argument("--n_client", help="number of clients", type=int)
    parser.add_argument("--n_round", help="number of federated training round", type=int)
    parser.add_argument("--eval_freq", help="evaluation interval", type=int, default=10)

    # SERVER OPTIMIZER
    parser.add_argument("--server_lr", help="learning rate in the server", type=float, default=1.)

    # LOCAL TRAINING
    parser.add_argument("--local_mini_batches", help="number of mini batches within local training", type=int, required=True)
    parser.add_argument("--local_update_epochs", help="number of local update steps", type=json.loads, required=True)
    parser.add_argument("--opt_type", help="type of optimizer", type=str, choices=["SGD", "Adam"], default="SGD")
    parser.add_argument("--lr", help="local learning rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", help="factor for l2 weight regularization", type=float, default=0.)
    parser.add_argument("--gauss_noise", help="scale of gaussian noise", type=float, required=True)

    # for sk-nsgd
    parser.add_argument("--tau", help="averaging interval tau", type=int, required=True)



    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    # fix random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load data
    train_loaders, test_loaders, n_in, n_out = get_dataloaders(
        data_type=args.data_type, n_client=args.n_client, batch_size=args.batch_size, num_workers=args.num_workers)

    # load model
    model = LR(n_in, n_out).to(device)

    # load optimizer class
    cls_opt = getattr(torch.optim, args.opt_type)

    # verify the length of local_update_epochs and n_round
    assert len(args.local_update_epochs) == args.n_round, f"The length of args.local_update_epochs {len(args.local_update_epochs)} should equal to args.n_round {args.n_round}"

    # -----------------------------------------
    # MAIN LOOP
    # -----------------------------------------
    # server model
    server_params = deepcopy(model.state_dict())

    for i in range(1, args.n_round + 1):

        logging.info("-" * 10 + f"The {i}-th training round starts" + "-" * 10)
        # Training
        avg_deltas, sum_samples = dict(), 0

        for client_id in range(args.n_client):
            # train from the server parameters
            model.load_state_dict(server_params)
            # local training
            client_params, n_samples, loss_avg = local_train(model=model,
                                                             dataloader=train_loaders[client_id],
                                                             # local update epochs in this round
                                                             local_update_epochs=args.local_update_epochs[i-1],
                                                             local_mini_batches=args.local_mini_batches,
                                                             optimizer=cls_opt,
                                                             lr=args.lr,
                                                             device=device,
                                                             weight_decay=args.weight_decay,
                                                             scale_noise=0,
                                                             tau=args.tau
                                                             )

            logging.info(f"Round #{i}, Client #{client_id}: Training loss {loss_avg}")

            with torch.no_grad():
                # calculate and scale delta
                scaled_deltas = {_: (client_params[_] - server_params[_]) for _ in
                                 server_params.keys()}
                # inject gauss noise into delta
                for key, delta in scaled_deltas.items():
                    scaled_deltas[key].data = delta.data + torch.normal(mean=0, std=args.gauss_noise, size=delta.size(),device=device)

                # online aggregate deltas
                for key in scaled_deltas.keys():
                    # new deltas from client
                    new_delta = scaled_deltas[key].data
                    old_avg_delta = avg_deltas.get(key, 0.)
                    avg_deltas[key] = old_avg_delta + (new_delta - old_avg_delta) * n_samples / (
                            sum_samples + n_samples)

            sum_samples += n_samples

        # Update the server parameters by the server learning rate
        with torch.no_grad():
            # scale back
            for key, delta in avg_deltas.items():
                avg_deltas[key] = delta
            # Update the global model
            for key, old_param in server_params.items():
                server_params[key] = old_param + args.server_lr * avg_deltas[key]

        # evaluate every ${eval_freq} rounds
        if i % args.eval_freq == 0:
            model.load_state_dict(server_params)
            # Evaluation
            sum_loss, sum_correct, sum_test_samples = 0., 0., 0.
            # for client_id in range(args.n_client):
            for test_loader in test_loaders:
                loss_client, n_correct_client, n_samples_client = local_test(model, test_loader, device)
                sum_loss += loss_client
                sum_correct += n_correct_client
                sum_test_samples += n_samples_client

            logging.info(
                f"Evaluation at Round #{i}: Test loss {sum_loss / sum_test_samples}, Test acc {sum_correct / sum_test_samples}, Test sample {sum_test_samples}")
