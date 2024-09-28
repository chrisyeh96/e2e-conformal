"""Battery storage data loader.

Split sizes
- train: 1400
- val: 351
- test: 438
"""
from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor  # for typing annotations
from torch.utils.data import DataLoader, TensorDataset

TRAIN_CALIB_FRAC = 0.8  # out of all data, first 80% for training+calibration, last 20% for test
TRAIN_FRAC = 0.8        # of the training+calib data, use a 80% train / 20% calib split


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Loads saved features and labels.

    Returns:
        X: shape [2189, 101], type float32, standardized features
        Y: shape [2189, 24], type float32, energy prices
        dates: shape [2189], type datetime64[D], dates of the prices
    """
    with np.load('storage/data/data.npz') as npz:
        X = npz['X']
        Y = npz['Y']
        dates = npz['dates']
    assert X.shape == (2189, 101)
    assert Y.shape == (2189, 24)
    return X.astype(np.float32), Y.astype(np.float32), dates.astype('datetime64[D]')


def get_traincalib_test_split(
    X: np.ndarray, Y: np.ndarray, dates: np.ndarray, shuffle: bool = False
) -> dict[str, Tensor | np.ndarray]:
    """Split data into traincalib and test sets.

    Args:
        X: shape [2189, 101], type float32, standardized features
        Y: shape [2189, 24], type float32, energy prices
        dates: shape [2189], type datetime64[D], dates of the prices
        shuffle: whether to shuffle the data before splitting.
            if False, test is always the last 20% of the data

    Returns:
        tensors: dict with keys '{X/Y/date}_traincalib', '{X/Y/date}_test'
    """
    if shuffle:
        rng = np.random.default_rng(seed=100)
        inds = rng.permutation(len(X))
        X = X[inds]
        Y = Y[inds]
        dates = dates[inds]

    n = int(X.shape[0] * TRAIN_CALIB_FRAC)
    tensors: dict[str, Tensor | np.ndarray] = {
        'X_traincalib': torch.from_numpy(X[:n]),
        'Y_traincalib': torch.from_numpy(Y[:n]),
        'X_test': torch.from_numpy(X[n:]),
        'Y_test': torch.from_numpy(Y[n:]),
        'date_traincalib': dates[:n],
        'date_test': dates[n:]
    }
    return tensors


def get_train_calib_split(
    tensors: Mapping[str, Tensor | np.ndarray], seed: int
) -> tuple[dict[str, Tensor], dict[str, np.ndarray]]:
    """Splits traincalib into train and calib splits.

    Args:
        tensors: dict with keys '{X/Y/date}_traincalib', '{X/Y/date}_test'
        seed: random seed for permuting traincalib before splitting.
            Does not change the test set.

    Returns:
        new_tensors: dict with keys '{X/Y}_{train/calib/test}'
        new_dates: dict with keys 'train', 'calib', 'test'
    """
    X_traincalib: Tensor = tensors['X_traincalib']  # type: ignore
    Y_traincalib: Tensor = tensors['Y_traincalib']  # type: ignore
    dates_traincalib: np.ndarray = tensors['date_traincalib']  # type: ignore

    rng = np.random.default_rng(seed=seed)
    inds = rng.permutation(len(X_traincalib))

    n_train = int(len(X_traincalib) * TRAIN_FRAC)
    train_inds = inds[:n_train]
    calib_inds = inds[n_train:]

    new_tensors: dict[str, Tensor] = {
        'X_train': X_traincalib[train_inds],
        'Y_train': Y_traincalib[train_inds],
        'X_calib': X_traincalib[calib_inds],
        'Y_calib': Y_traincalib[calib_inds],
        'X_test': tensors['X_test'],  # type: ignore
        'Y_test': tensors['Y_test'],  # type: ignore
    }
    new_dates: dict[str, np.ndarray] = {
        'train': dates_traincalib[train_inds],
        'calib': dates_traincalib[calib_inds],
        'test': tensors['date_test'],  # type: ignore
    }
    return new_tensors, new_dates


def get_loaders(tensors: Mapping[str, Tensor], batch_size: int) -> dict[str, DataLoader]:
    """
    Args:
        tensors: dict with keys '{X/Y}_{train/calib/test}'
        batch_size: only applied to train_loader. Set to -1 to load all training data in
            1 batch. Calib and Test sets are small enough that we should always load
            them all at once.

    Returns:
        loaders: maps split (train, calib, test) to DataLoader
    """
    shuffle = True
    if batch_size == -1:
        batch_size = len(tensors['X_train'])
        shuffle = False
    train_loader = DataLoader(
        TensorDataset(tensors['X_train'], tensors['Y_train']),
        shuffle=shuffle, batch_size=batch_size, pin_memory=True)

    num_test = len(tensors['X_test'])
    test_loader = DataLoader(
        TensorDataset(tensors['X_test'], tensors['Y_test']),
        shuffle=False, batch_size=num_test, pin_memory=True)

    num_calib = len(tensors['X_calib'])
    calib_loader = DataLoader(
        TensorDataset(tensors['X_calib'], tensors['Y_calib']),
        shuffle=False, batch_size=num_calib, pin_memory=True)
    return {'train': train_loader, 'test': test_loader, 'calib': calib_loader}


def get_tensors(
    shuffle: bool, log_prices: bool
) -> tuple[dict[str, Tensor | np.ndarray], str | tuple[np.ndarray, np.ndarray]]:
    """Gets tensors for traincalib and test splits, with transformed price labels.

    Args:
        shuffle: whether to shuffle the data before splitting into traincalib / test.
            if False, test is always the last 20% of the data
        log_prices: whether to log-transform the prices,
            otherwise standardizes prices.

    Returns:
        tensors: dict with keys '{X/Y/date}_traincalib', '{X/Y/date}_test'
        y_info: either 'log' or a tuple (Y_mean, Y_std) for unstandardizing
    """
    # Load data
    X, Y, dates = load_data()
    y_info: str | tuple[np.ndarray, np.ndarray]
    if log_prices:
        Y = np.log(Y)
        y_info = 'log'
    else:
        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0)
        Y = (Y - Y_mean) / Y_std
        y_info = (Y_mean, Y_std)
    tensors = get_traincalib_test_split(X, Y, dates, shuffle=shuffle)
    return tensors, y_info
