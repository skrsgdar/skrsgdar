import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", help="which device to run on, choice from ['CPU', '${INDEX_OF_GPU}']", type=str)
    parser.add_argument("--seed", help="random seed", type=int, default=1234)

    # DATA
    parser.add_argument("--batch_size", help="batch size of the data", type=int)
    parser.add_argument("--data_type", help="which data to use", type=str, choices=["ADULT", "BANK", "IPUMS-US", "IPUMS-BR", "CANCER", "CREDIT", "MAGIC",  "CLIENT"])
    parser.add_argument("--num_workers", type=int, default=8)

    # FEDERATED TRAINING
    parser.add_argument("--n_client", help="number of clients", type=int)
    parser.add_argument("--n_round", help="number of federated training round", type=int)

    # SERVER OPTIMIZER
    parser.add_argument("--server_lr", help="learning rate in the server", type=float, default=1.)

    # LOCAL TRAINING
    parser.add_argument("--local_update_steps", help="number of local update steps", type=int, default=1)
    parser.add_argument("--opt_type", help="type of optimizer", type=str, choices=["SGD", "Adam"], default="SGD")
    parser.add_argument("--lr", help="local learning rate", type=float, default=0.01)
    parser.add_argument("--eval_freq", help="evaluation interval", type=int, default=10)

    # LOCAL TRAINING NOISE
    parser.add_argument("--perturb_grad", help="inject noise into gradient during local training or not", type=bool, default=False)
    parser.add_argument("--scale_grad", help="scale of local gaussian noise", type=float, default=0.01)

    # GLOBAL NOISE BEFORE TEST
    parser.add_argument("--perturb_weight", help="inject noise into weight after local training or not", type=bool, default=False)
    parser.add_argument("--scale_weight", help="scale of global gaussian noise", type=float, default=0.01)

    # for sk-nsgd
    parser.add_argument("--scale_parameter", type=float, required=True)
    parser.add_argument("--possion_rate", type=float, required=True)
    parser.add_argument("--l2_sensitivity", type=float, required=True)

    args = parser.parse_args()

    return args


