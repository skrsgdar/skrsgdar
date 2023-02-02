import sys

sys.path.append('..')

from utils import criterion, clip_grad_by_l2norm, discrete_tensor, skellam_noise
from model import LR
from copy import deepcopy
import random
import numpy as np
import torch

from builder import get_dataloaders

import argparse
import logging

logging.getLogger().setLevel(logging.INFO)


def local_train(model,
                dataloader,
                optimizer,
                lr,
                grad_clip,
                device,
                scale_parameter,
                possion_rate,
                l2_sensitivity,
                weight_decay):
    model.train()
    # init optimizer
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)

    id_batch, n_samples, loss_total = 0, 0., 0.

    # only one batch
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

        with torch.no_grad():
            # clip and discrete
            for param in model.parameters():
                # gradient for each sample
                for i, grad_sample in enumerate(param.grad_sample):
                    # clip gradient for each sample
                    param.grad_sample[i] = clip_grad_by_l2norm(grad_sample, grad_clip) * scale_parameter
                    # discrete gradient for each sample
                    param.grad_sample[i] = discrete_tensor(param.grad_sample[i], l2_sensitivity)

                # aggregate gradients and inject noise
                param.grad.data = torch.sum(param.grad_sample, dim=0) + skellam_noise(param.size(), possion_rate, param.device)

            n_samples += len(y)
            loss_total += loss.item() * len(y)

            for param in model.parameters():
                param.grad_sample = None

        # only one batch
        break

    return {name: param.grad.data for name, param in model.named_parameters()}, n_samples, loss_total / n_samples

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
    parser.add_argument("--data_type", help="which data to use", type=str,
                        choices=["ADULT", "BANK", "IPUMS-US", "IPUMS-BR", "CANCER", "CREDIT", "MAGIC",  "CLIENT"])
    parser.add_argument("--q", help="subsampling rate", type=float, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_freq", help="evaluation interval", type=int, default=10)

    # FEDERATED TRAINING
    parser.add_argument("--n_client", help="number of clients", type=int)
    parser.add_argument("--n_round", help="number of federated training round", type=int)

    # SERVER OPTIMIZER
    parser.add_argument("--server_lr", help="learning rate in the server", type=float, default=1.)

    # LOCAL TRAINING
    parser.add_argument("--opt_type", help="type of optimizer", type=str, choices=["SGD", "Adam"], default="SGD")
    parser.add_argument("--lr", help="local learning rate", type=float, default=0.01)
    parser.add_argument("--weight_decay", help="factor for l2 weight regularization", type=float, default=0.)

    # LOCAL TRAINING NOISE
    parser.add_argument("--grad_clip", help="threshold for gradient clipping (-1 means not clip)", default=-1,
                        type=float)
    parser.add_argument("--scale_parameter", type=float, required=True)
    parser.add_argument("--possion_rate", help="possion rate for sk noise", type=float, default=0.01)
    parser.add_argument("--l2_sensitivity", type=float, required=True)

    args = parser.parse_args()

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    # fix random seed
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # load data
    train_loaders, test_loaders, n_in, n_out = get_dataloaders(
        data_type=args.data_type, n_client=args.n_client, q=args.q, batch_size=None, num_workers=args.num_workers)

    # load model
    model = LR(n_in, n_out).to(device)

    # load optimizer class
    cls_opt = getattr(torch.optim, args.opt_type)

    # -----------------------------------------
    # MAIN LOOP
    # -----------------------------------------
    # server model
    server_params = deepcopy(model.state_dict())

    # The actual number of federated training rounds
    n_round = int(args.n_round / args.q)

#     logging.info(f"The actual training round number is {n_round}")

    for i in range(1, n_round + 1):

#         logging.info("-" * 10 + f"The {i}/{n_round}-th training round starts" + "-" * 10)

        avg_gradient, sum_samples = dict(), 0.
        for client_id in range(args.n_client):
            # train from the server parameters
            model.load_state_dict(server_params)

            sum_gradient, n_samples, loss_avg = local_train(model=model,
                                                            dataloader=train_loaders[client_id],
                                                            optimizer=cls_opt,
                                                            lr=args.lr,
                                                            device=device,
                                                            grad_clip=args.grad_clip,
                                                            scale_parameter=args.scale_parameter,
                                                            possion_rate=args.possion_rate,
                                                            l2_sensitivity=args.l2_sensitivity,
                                                            weight_decay=args.weight_decay)

#             logging.info(f"Round #{i}, Client #{client_id}: Training loss {loss_avg} with {n_samples} samples")

            # record average gradient and number of samples
            with torch.no_grad():
                for key, new_sum_grad in sum_gradient.items():
                    old_avg_grad = avg_gradient.get(key, 0.)
                    avg_gradient[key] = old_avg_grad + (new_sum_grad - old_avg_grad * n_samples) / (sum_samples + n_samples)

                sum_samples += n_samples

        with torch.no_grad():
            # retrieve gradient and model update
            for key, value in server_params.items():
                server_params[key] -= args.server_lr * avg_gradient[key] / float(args.scale_parameter)

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
            
#             logging.info(f"{sum_correct / sum_test_samples},")
            logging.info(
                f"Evaluation at Round #{i}: Test loss {sum_loss / sum_test_samples}, Test acc {sum_correct / sum_test_samples}, Test sample {sum_test_samples}")
