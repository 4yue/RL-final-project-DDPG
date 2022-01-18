import torch
from torch.autograd import Variable

FLOAT = torch.FloatTensor


def copy_para(target, source):
    for i, j in zip(target.parameters(), source.parameters()):
        i.data.copy_(j)


def to_tensor(ndarray, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), requires_grad=requires_grad
    ).type(dtype)


def to_numpy(variable):
    return variable.data.numpy()
