from itertools import repeat
import numpy as np
import random
import torch


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def set_seed(SEED=777):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
