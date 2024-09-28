import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset


def train_test_scaler(train, val, test):
    scaler = StandardScaler()
    normalized_x_train = pd.DataFrame(scaler.fit_transform(train))
    normalized_x_val = pd.DataFrame(scaler.transform(val))
    normalized_x_test = pd.DataFrame(scaler.transform(test))
    return normalized_x_train, normalized_x_val, normalized_x_test


def reshuffle(train, val, test, perm):
    a, b, c = len(train), len(val), len(test)
    assert len(perm) == a + b + c
    combined = pd.concat([train, val, test], axis='index')
    train = combined.iloc[perm[:a]]
    val = combined.iloc[perm[a:a+b]]
    test = combined.iloc[perm[a+b:a+b+c]]
    return train, val, test

def portfolio_data_gen(year, seed, shuffled):
    year_directory = f'portfolio/data/yfinance/{year}_samples'
    returns_file = f'{year}_returns_{0}.txt'
    side_file = f'{year}_data_side_{0}.txt'

    returns_file_path = os.path.join(year_directory, returns_file)
    side_file_path = os.path.join(year_directory, side_file)

    with open(returns_file_path, 'r') as f:
        lines = f.readlines()
        returns_lists = [list( line.strip().split(',')) for line in lines]

    with open(side_file_path, 'r') as f:
        lines = f.readlines()
        side_info_lists = [list( line.strip().split(',')) for line in lines]

    returns_cols = returns_lists[seed]
    side_info_cols = side_info_lists[seed]

    returns = pd.read_csv('portfolio/data/yfinance/expected_return.csv')
    data_side = pd.read_csv('portfolio/data/yfinance/side_info.csv')

    def create_train_val_test(df, year, col_list):

        df_sub = df[col_list].copy()
        df_sub['DATE'] = pd.to_datetime(df_sub['DATE'])
        df_sub['year'] = df_sub['DATE'].dt.year
        df_sub['year'] = df_sub['year'].astype(int)

        # Parse start year to integer
        start_year = int(year)
        train_end_year = start_year + 2
        val_start_year = train_end_year+1
        test_start_year= val_start_year+1

        df_train = df_sub[(df_sub.year >= start_year) & (df_sub.year <= train_end_year)].copy()
        df_val = df_sub[df_sub.year == val_start_year].copy()
        df_test = df_sub[df_sub.year == test_start_year].copy()

        df_train = df_train.drop(['DATE', 'year'], axis=1)
        df_val = df_val.drop(['DATE', 'year'], axis=1)
        df_test = df_test.drop(['DATE', 'year'], axis=1)
        return df_train, df_val, df_test


    returns_cols += ['DATE']
    side_info_cols += ['DATE']
    returns_train, returns_val, returns_test = create_train_val_test(returns, year, returns_cols)
    side_train, side_val, side_test = create_train_val_test(data_side, year, side_info_cols)

    if shuffled:
        rng = np.random.default_rng(seed=seed)
        perm = rng.permutation(len(returns_train) + len(returns_val) + len(returns_test))
        returns_train, returns_val, returns_test = reshuffle(returns_train, returns_val, returns_test, perm)
        side_train, side_val, side_test = reshuffle(side_train, side_val, side_test, perm)

    returns_means = returns_train.mean()
    returns_train.fillna(returns_means, inplace=True)
    returns_val.fillna(returns_means, inplace=True)
    returns_test.fillna(returns_means, inplace=True)
    side_means = side_train.mean()
    side_train.fillna(side_means, inplace=True)
    side_val.fillna(side_means, inplace=True)
    side_test.fillna(side_means, inplace=True)

    # returns_train_s, returns_val_s, returns_test_s = train_test_scaler(returns_train, returns_val, returns_test) # don't double standardize!

    returns_train_s = torch.from_numpy(100*returns_train.values)
    returns_val_s   = torch.from_numpy(100*returns_val.values)
    returns_test_s  = torch.from_numpy(100*returns_test.values)

    side_train_s, side_val_s, side_test_s = train_test_scaler(side_train, side_val, side_test)
    side_train_s, side_val_s, side_test_s = torch.from_numpy(side_train_s.values), torch.from_numpy(side_val_s.values), torch.from_numpy(side_test_s.values)

    return side_train_s, side_val_s, side_test_s, returns_train_s, returns_val_s, returns_test_s


def get_loaders(
    batch_size: int, year: int, seed: int, shuffled: bool
) -> tuple[dict[str, DataLoader], tuple[np.ndarray, np.ndarray]]:
    """
    Args:
        batch_size
        year: the starting year in [2012, 2013, 2014].
        seed: the randomizing seed as in the e2e-cro paper
        shuffled: whether to shuffle the data before doing train-test split (losing casality)

    Returns:
        loaders_dict: dict mapping split ('train', 'calib', 'test') to DataLoader
        y_info: tuple of arrays (y_mean, y_std)

    Dimensions:
        x: 21
        y: 15

    Train has 252*3 samples, calib has 252, and test has 252.

    Note: x is standardized with sklearn StandardScaler.
    """
    X_train, X_cal, X_test, y_train, y_cal, y_test = portfolio_data_gen(year, seed, shuffled)
    X_train = X_train.to(torch.float32)
    X_cal = X_cal.to(torch.float32)
    X_test = X_test.to(torch.float32)
    y_train = y_train.to(torch.float32)
    y_cal = y_cal.to(torch.float32)
    y_test = y_test.to(torch.float32)

    with torch.no_grad():
        all_y = torch.cat([y_train, y_cal, y_test], dim=0)
        y_mean = all_y.mean(dim=0)
        y_std = all_y.std(dim=0)
        y_info = (y_mean.numpy(), y_std.numpy())
        
        y_train = (y_train - y_mean) / y_std
        y_cal = (y_cal - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

    train_loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)
    calib_loader = DataLoader(TensorDataset(X_cal, y_cal), shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), shuffle=False, batch_size=batch_size)
    loaders_dict = {'train': train_loader, 'test': test_loader, 'calib': calib_loader}

    return loaders_dict, y_info
