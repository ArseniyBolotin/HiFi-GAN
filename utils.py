from itertools import repeat


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader
