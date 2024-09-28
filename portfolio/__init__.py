import numpy as np
import torch.utils.data

from portfolio import synthetic, yfinance


def get_loaders(
    dataset: str, batch_size: int, seed: int, shuffle: bool
) -> tuple[dict[str, torch.utils.data.DataLoader], tuple[np.ndarray, np.ndarray]]:
    if dataset == 'synthetic':
        alpha = 0.9
        phi = 0.7
        loaders, y_info = synthetic.get_loaders(
            batch_size, seed=seed, alpha=alpha, phi=phi)
    elif dataset == 'yfinance':
        loaders, y_info = yfinance.get_loaders(
            batch_size, year=2013, seed=seed, shuffled=shuffle)
    else:
        raise NotImplementedError
    return loaders, y_info
