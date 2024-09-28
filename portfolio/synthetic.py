import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def generate_mixture_data(mean_list,cov_list, prob_list, num_samples):
    # TODO: this function is poorly written, coule be simplified with one line of scipy
    mixK = len(prob_list)

    mix_rand_u = np.random.rand(num_samples)
    data = np.random.multivariate_normal(mean_list[0], cov_list[0], 0)

    sumPs = 0

    for mix_index in range(mixK):
        data_a = np.random.multivariate_normal(mean_list[mix_index], cov_list[mix_index], num_samples)

        sumPs = sumPs+prob_list[mix_index]
        mix_N = (mix_rand_u<=sumPs).sum()-len(data)
        data = np.concatenate((data, data_a[:mix_N,:]), axis=0)

    np.random.shuffle(data)
    return data


def generate_covariance_matrix(variance_1, variance_2, correlation, spread_factor):
    # Create the 2x2 covariance matrix for the independent features
    cov_independent = np.array([[variance_1, 0],
                               [0, variance_2]])

    # Create the 2x2 submatrix with desired positive semi-definite properties
    submatrix = np.array([[variance_1, correlation * np.sqrt(variance_1 * variance_2)],
                          [correlation * np.sqrt(variance_1 * variance_2), variance_2]])

    # Create the 4x4 covariance matrix
    cov_matrix = np.zeros((4, 4))
    cov_matrix[:2, :2] = cov_independent
    cov_matrix[2:4, 2:4] = submatrix * spread_factor
    cov_matrix[0, 2] = correlation * np.sqrt(variance_1 * variance_2)
    cov_matrix[2, 0] = correlation * np.sqrt(variance_1 * variance_2)

    return cov_matrix


def compute_train_test_split(X, Y):
    # TODO: also badly written function. could simplify
    x_train = np.array(Y.iloc[:600, :])
    x_val   = np.array(Y.iloc[600:1000, :])
    x_test  = np.array(Y.iloc[1000:2000, :])
    y_train = np.array(X.iloc[:600, :])
    y_val   = np.array(X.iloc[600:1000,:])
    y_test  = np.array(X.iloc[1000:2000, :])
    # print(x_train.shape,x_val.shape, x_test.shape, y_train.shape,y_val.shape, y_test.shape)
    return (torch.from_numpy(x_train), torch.from_numpy(x_val), torch.from_numpy(x_test),
            torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test),)


def create_gmm(alpha: float, phi: float) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[float, float, float]
]:
    # Parameters
    mean_a = np.array([0, 0, 0, 0])
    mean_b = np.array([0, 5, 5, 0])
    mean_c = mean_b

    variance_1 = 1.0
    variance_2 = 1.5
    correlation = 0.3
    spread_factor = 2.0
    cov_a = generate_covariance_matrix(variance_1, variance_2, correlation, spread_factor)

    # Varying values of alpha
    p_a = phi
    p_b = (1 - p_a) / (alpha + 1)
    p_c = (1 - p_a) * alpha / (alpha + 1)

    cov_b = cov_a * alpha
    cov_c = cov_a / alpha

    means = (mean_a, mean_b, mean_c)
    covs = (cov_a, cov_b, cov_c)
    probs = (p_a, p_b, p_c)
    return means, covs, probs


def data_gen(alpha, phi, seed=123):
    """alpha, phi are parameters of the weights of three mixtures (`prob_list`)
    returns X_train, X_val, X_test, y_train, y_val, y_test
    """
    means, covs, probs = create_gmm(alpha, phi)

    np.random.seed(seed)
    num_samples = 2000
    data = generate_mixture_data(means, covs, probs, num_samples)

    df = pd.DataFrame(data[:, :2])
    aux1 = pd.DataFrame(data[:, 2:])

    x_train, x_val, x_test, y_train, y_val, y_test = compute_train_test_split(df, aux1)
    x_scaler, x_val_scaler, x_test_scaler = x_train, x_val, x_test #train_test_scaler(x_train, x_val, x_test)
    return x_scaler, x_val_scaler, x_test_scaler, y_train, y_val, y_test


def get_loaders(
    batch_size: int, seed: int, alpha: float = 0.1, phi: float = 0.1
) -> tuple[dict[str, DataLoader], tuple[np.ndarray, np.ndarray]]:
    """
    Dimensions:
        x: 2
        y: 2

    Train has 600 samples, calib has 400, and test has 1000.

    Args:
        batch_size: batch size for dataloaders
        alpha, phi: parameters for the weights of 3 4D multivar normals
        seed: the randomizing seed as in the e2e-cro paper

    Returns:
        loaders_dict: dict mapping split ('train', 'calib', 'test') to DataLoader
        y_info: tuple of Tensors (y_mean, y_std)
    """
    X_train, X_cal, X_test, y_train, y_cal, y_test = data_gen(alpha, phi, seed)
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
