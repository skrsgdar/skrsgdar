import sys

sys.path.append('..')

import argparse
import torch
import numpy as np
import random

from utils import criterion
from utils import clip_grad_by_l2norm
from builder import get_dataloaders
from model import LR

import logging

logging.getLogger().setLevel(logging.INFO)


def str2bool(v):
    if v.lower() in ['yes', 'true']:
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", help="which device to run on, choice from ['CPU', '${INDEX_OF_GPU}']", type=str)
    parser.add_argument("--seed", help="random seed", type=int, default=-1)

    # DATA
    parser.add_argument("--batch_size", help="batch size of the data", type=int)
    parser.add_argument("--data_type", help="which data to use", type=str,
                        choices=["ADULT", "BANK", "IPUMS-US", "IPUMS-BR", "CANCER", "CREDIT", "MAGIC",  "CLIENT"])
    parser.add_argument("--num_workers", type=int, default=1)

    # TRAIN
    parser.add_argument("--n_epoch", help="", type=int)
    parser.add_argument("--opt_type", help="", type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_decay", help="if lr decay by the number of epochs", type=str2bool, default=False)
    parser.add_argument("--grad_clip", help="threshold for gradient clipping(not clip if <=0)", type=float, default=-1)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--eval_freq", type=int, default=10)

    # LOCAL TRAINING NOISE
    parser.add_argument("--perturb_grad", help="inject noise into gradient during local training or not", type=str2bool,
                        default=False)
    parser.add_argument("--scale_grad", help="scale of local gaussian noise", type=float, default=0.01)

    # GLOBAL NOISE BEFORE TEST
    parser.add_argument("--perturb_weight", help="inject noise into weight after local training or not", type=str2bool,
                        default=False)
    parser.add_argument("--scale_weight", help="scale of global gaussian noise", type=float, default=0.01)

    args = parser.parse_args()

    print(args)

    # fix random seed
    if args.seed != -1:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # get device
    if args.device.lower() == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}")

    # load data
    train_loaders, test_loaders, n_in, n_out = get_dataloaders(
        args.data_type,
        n_client=1,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    train_loader, test_loader = train_loaders[0], test_loaders[0]

    model = LR(n_in, n_out).to(device)

    optimizer = getattr(torch.optim, args.opt_type)(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # -----------------------------------------
    # MAIN LOOP
    # -----------------------------------------
    for epoch_i in range(1, args.n_epoch + 1):
        # lr decay
        if args.lr_decay:
            for g in optimizer.param_groups:
                g['lr'] = args.lr / epoch_i

        model.train()
        sum_loss = 0.
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # forward
            pred = model(x)
            if len(y.size()) == 1:
                y = torch.unsqueeze(y, dim=-1)
            loss = criterion(pred, y)

            # backward
            optimizer.zero_grad()
            loss.backward()

            # clip
            if args.grad_clip > 0:
                for param in model.parameters():
                    # clip each sample
                    for i, grad_sample in enumerate(param.grad_sample):
                        param.grad_sample[i] = clip_grad_by_l2norm(grad_sample, args.grad_clip)
                    # aggregate gradients
                    param.grad.data = torch.mean(param.grad_sample, dim=0)


            # inject noise
            if args.perturb_grad:
                for param in model.parameters():
                    param.grad += 1. / float(len(y)) * torch.normal(mean=0, std=args.scale_grad, size=param.size(),
                                                                    device=param.device)

            optimizer.step()

            # clear gradient samples to avoid memory leakage
            for param in model.parameters():
                param.grad_sample = None

            sum_loss += loss.item() * len(y)

        if epoch_i % 10 == 0:
            logging.info(f"Round #{epoch_i}: Training loss {sum_loss / len(train_loader.dataset)}")

        if epoch_i % args.eval_freq == 0:
            with torch.no_grad():
                model.eval()
                sum_loss, sum_correct = 0., 0.
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)

                    pred = model(x)
                    if len(y.size()) == 1:
                        y = torch.unsqueeze(y, dim=-1)
                    loss = criterion(pred, y)

                    predict = 1. / (1. + torch.exp(- pred))

                    sum_correct += torch.sum(torch.sign(predict - 0.5) * (y - 0.5) * 2 == 1).item()
                    sum_loss += loss.item() * len(y)

                if epoch_i == args.n_epoch:
                    logging.info(
                        f"Final evaluation at Round #{epoch_i}: Test loss {sum_loss / len(test_loader.dataset)}, Test acc {sum_correct / len(test_loader.dataset)}")
                else:
                    logging.info(
                        f"Evaluation at Round #{epoch_i}: Test loss {sum_loss / len(test_loader.dataset)}, Test acc {sum_correct / len(test_loader.dataset)}")

    if args.perturb_weight:
        with torch.no_grad():
            for param in model.parameters():
                param.data += torch.normal(mean=0., std=args.scale_weight, size=param.size(), device=param.device)

            model.eval()
            sum_loss, sum_correct = 0., 0.
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                if len(y.size()) == 1:
                    y = torch.unsqueeze(y, dim=-1)
                loss = criterion(pred, y)

                predict = 1. / (1. + torch.exp(- pred))

                sum_correct += torch.sum(torch.sign(predict - 0.5) * (y - 0.5) * 2 == 1).item()
                sum_loss += loss.item() * len(y)

            logging.info(
                f"Final evaluation with weight perturbation {args.scale_weight}: Test loss {sum_loss / len(test_loader.dataset)}, Test acc {sum_correct / len(test_loader.dataset)}")
