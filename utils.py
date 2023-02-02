
import torch

def least_power_2_upper_bound(d):
        upper_bound = 1
        while upper_bound < d:
            upper_bound = upper_bound * 2
        return upper_bound

def discrete_tensor(tensor, l2_sensitivity):
    floor = torch.floor(tensor)
    p_ceil = tensor - floor
    while True:
        random_nums = torch.rand(size=tensor.size(), device=tensor.device, requires_grad=False)
        choice_floor = (random_nums > p_ceil).type(torch.float32)

        discrete_tensor = choice_floor * floor + (1. - choice_floor) * (floor + 1.)

        l2_norm_square = torch.norm(discrete_tensor, p=2) ** 2

        if l2_norm_square <= l2_sensitivity ** 2:
            return discrete_tensor

    raise ValueError("Cannot discrete tensor")

def discrete_tensor_extra(tensor, extra_in_l2_squared):
    original_norm = torch.norm(tensor, p=2)
    floor = torch.floor(tensor)
    p_ceil = tensor - floor
    while True:
        random_nums = torch.rand(size=tensor.size(), device=tensor.device, requires_grad=False)
        choice_floor = (random_nums > p_ceil).type(torch.float32)

        discrete_tensor = choice_floor * floor + (1. - choice_floor) * (floor + 1.)

        l2_norm_square = torch.norm(discrete_tensor, p=2) ** 2

        if l2_norm_square <= original_norm ** 2 + extra_in_l2_squared:
            return discrete_tensor

    raise ValueError("Cannot discrete tensor")

def discrete_tensor_l2(tensor, beta):
    original_norm = torch.norm(tensor, p=2)
    floor = torch.floor(tensor)
    p_ceil = tensor - floor
    while True:
        random_nums = torch.rand(size=tensor.size(), device=tensor.device, requires_grad=False)
        choice_floor = (random_nums > p_ceil).type(torch.float32)

        discrete_tensor = choice_floor * floor + (1. - choice_floor) * (floor + 1.)

        l2_norm_square = torch.norm(discrete_tensor, p=2) ** 2

        if torch.norm(torch.subtract(tensor,discrete_tensor), p=2) <= beta * float(tensor.size()[0])** 0.5:
            return discrete_tensor

    raise ValueError("Cannot discrete tensor")

def clip_grad_by_l2norm(grad, threshold):
    grad_norm = torch.norm(grad, p=2)
    return grad * (min(grad_norm, threshold) / grad_norm)


def skellam_noise(size, lambda_, device):
    noise = torch.poisson(torch.ones(size=size) * lambda_) - torch.poisson(torch.ones(size=size) * lambda_)
    return noise.to(device)


# loss for logistic regression
def criterion(pred, label):
    return torch.mean(torch.log(1. + torch.pow(torch.exp(-1. * pred), 2. * label - 1.)))


def compute_per_sample_grad(model, x, label, criterion):
    x = torch.unsqueeze(x)
    label = torch.unsqueeze(label)

    pred = model(x)
    loss = criterion(pred, label)

    return torch.autograd.grad(loss, list(model.parameters))

def str2bool(v):
    if v.lower() in ['yes', 'true']:
        return True
    else:
        return False
