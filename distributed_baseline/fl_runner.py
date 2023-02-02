from model import LR
from parser import parse_args
from copy import deepcopy
from utils import criterion
import random
import numpy as np
import torch

from builder import get_dataloaders

import logging

logging.getLogger().setLevel(logging.INFO)


def local_train(model,
                dataloader,
                local_update_steps,
                optimizer,
                lr,
                device):
    model.train()
    # init optimizer
    opt = optimizer(model.parameters(), lr=lr)

    id_batch, n_samples, loss_total = 0, 0., 0.

    while id_batch < local_update_steps:
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

            # clip grads
            if args.grad_clip_method == "L2":
                for param in model.parameters():
                    # TODO: clipping method
                    pass

            # inject noise
            if args.perturb_grad:
                for param in model.parameters():
                    param.grad += torch.normal(0, std=args.scale_grad, size=param.size(), device=param.device)

            opt.step()

            n_samples += len(x)
            loss_total += loss.item()

            id_batch += 1

    # inject noise into weight after local training
    if args.perturb_weight:
        for param in model.parameters():
            param.grad += torch.normal(0, std=args.scale_weight, size=param.size(), device=param.device)

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
            loss_total += loss.item()

    return loss_total, n_correct, len(dataloader.dataset)


if __name__ == '__main__':
    args = parse_args()

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

    # load criterion
    def criterion(pred, label):
        return torch.mean(torch.log(1. + torch.pow(torch.exp(-1. * pred), 2. * label - 1.)))

    # -----------------------------------------
    # MAIN LOOP
    # -----------------------------------------
    # server model
    server_params = deepcopy(model.state_dict())

    for i in range(1, args.n_round+1):

        logging.info("-" * 10 + f"The {i}-th training round starts" + "-" * 10)
        # Training
        avg_params, sum_samples = dict(), 0
        for client_id in range(args.n_client):
            # train from the server parameters
            model.load_state_dict(server_params)
            # local training
            client_params, n_samples, loss_avg = local_train(model,
                                                             train_loaders[client_id],
                                                             args.local_update_steps,
                                                             cls_opt,
                                                             lr=args.lr,
                                                             device=device)

            logging.info(f"Round #{i}, Client #{client_id}: Training loss {loss_avg}")

            # online aggregation
            for key in client_params.keys():
                # new params from client
                new_param = client_params[key]
                old_avg_param = avg_params.get(key, 0.)
                avg_params[key] = old_avg_param + (new_param - old_avg_param) * n_samples / (sum_samples + n_samples)

            sum_samples += n_samples

        # Update the server parameters by the server learning rate
        with torch.no_grad():
            for key, old_param in server_params.items():
                new_param = avg_params[key]
                server_params[key] = old_param - args.server_lr * (old_param - new_param)

        # evaluate every ${eval_freq} rounds
        if i % args.eval_freq == 0:
            # Evaluation
            sum_loss, sum_correct, sum_test_samples = 0., 0., 0.
            # for client_id in range(args.n_client):
            for test_loader in test_loaders:
                loss_client, n_correct_client, n_samples_client = local_test(model, test_loader, device)
                sum_loss += loss_client
                sum_correct += n_correct_client
                sum_test_samples += n_samples_client

            logging.info(f"Evaluation at Round #{i}: Test loss {sum_loss / sum_test_samples}, Test acc {sum_correct / sum_test_samples}, Test sample {sum_test_samples}")
