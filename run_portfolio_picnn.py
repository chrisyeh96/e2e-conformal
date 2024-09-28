import argparse
from collections.abc import Collection
import functools
import itertools
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from models.picnn import fit_picnn_FALCON, PICNN
from models.picnn_e2e import optimize_wrapper, train_e2e
from portfolio import get_loaders
from portfolio.problems import PortfolioProblemPICNN
from utils.parse_args import parse_args
from utils.multiprocess import run_parallel


INPUT_DIM = {
    'synthetic': 2,
    'yfinance': 21
}
Y_DIM = {
    'synthetic': 2,
    'yfinance': 15
}
ETO_EPOCHS = 50
MAX_E2E_EPOCHS = 100
BATCH_SIZE = 256
SEEDS = range(10)

EPSILON = 0
Y_IN_OUTPUT_LAYER = False
N_LAYERS = 2
HIDDEN_DIM = 128
LARGE_Q_PENALTY = 1e-2

MALA_PARAMS: dict[str, Any] = dict(
    num_chains=50,
    num_samples=1,
    burnin=100,
    std=0.25,
    warm_start=False
)


def optimize_single(
    seed: int, ds: str, shuffle: bool, alpha: float, n_layers: int, hidden_dim: int,
    basename: str, out_dir: str
) -> tuple[dict[str, Any], str]:
    """Run optimize step for a single seed from a saved model.

    Always on CPU.

    Returns:
        result: dict of performance metrics
        msg: message string
    """
    loaders, (y_mean, y_std) = get_loaders(
        ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)
    prob = PortfolioProblemPICNN(
        N=Y_DIM[ds], L=n_layers, d=hidden_dim, y_mean=y_mean, y_std=y_std,
        epsilon=EPSILON)

    # load saved model
    model = PICNN(input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds], hidden_dim=hidden_dim,
                  n_layers=n_layers, y_in_output_layer=Y_IN_OUTPUT_LAYER, epsilon=EPSILON)
    ckpt_path = os.path.join(out_dir, f'{basename}_seed{seed}.pt')
    tqdm.write(f'Loading saved model from {ckpt_path}')
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    # then optimize
    result = optimize_wrapper(model, prob=prob, loaders=loaders, alpha=alpha)
    result['seed'] = seed

    msg = (f'alpha {alpha:.2g} train task loss {result["train_task_loss"]:.3f}, '
           f'test task loss {result["test_task_loss"]:.3f}, '
           f'test coverage {result["test_coverage"]:.2f}')
    return result, msg


def estimate_single(
    seed: int, ds: str, shuffle: bool, lr: float, l2reg: float, n_layers: int,
    hidden_dim: int, basename: str, out_dir: str
) -> tuple[dict[str, Any], str]:
    """
    Always on CPU.

    Returns:
        result: dict of performance metrics
        msg: message string
    """
    loaders, _ = get_loaders(ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)
    result: dict[str, Any] = {'seed': seed}

    model = PICNN(input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds], hidden_dim=hidden_dim,
                  n_layers=n_layers, y_in_output_layer=Y_IN_OUTPUT_LAYER, epsilon=EPSILON)
    model, train_losses = fit_picnn_FALCON(
        model, lr=lr, l2reg=l2reg, epochs=ETO_EPOCHS, loader=loaders['train'],
        weight_towards_zero=1., **MALA_PARAMS)
    result['train_losses'] = train_losses

    ckpt_path = os.path.join(out_dir, f'{basename}_seed{seed}.pt')
    torch.save(model.state_dict(), ckpt_path)

    return result, ''


def eto(
    ds: str, shuffle: bool, alphas: Collection[float], lr: float, l2reg: float,
    n_layers: int, hidden_dim: int, out_dir: str, multiprocess: int = 1
) -> None:
    """
    Saves a CSV file with columns:
        seed, train_losses, {train/test}_task_loss, {train/test}_coverage, q

    Args:
        ds: dataset, either 'synthetic' or 'yfinance'
        alphas: risk levels in (0, 1), the uncertainty set is a (1-alpha)-confidence set
        lr: learning rate
        l2reg: L2 regularization strength
        n_layers: # of hidden layers in PICNN
        hidden_dim: # of units per hidden layer
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        out_dir: where to save the results
        multiprocess: # of seeds to train in parallel
    """
    L = n_layers
    d = hidden_dim
    basename = f'eto_L{L}_d{d}_lr{lr:.3g}_reg{l2reg:.3g}'

    # estimate
    func = functools.partial(
        estimate_single, ds=ds, shuffle=shuffle, lr=lr, l2reg=l2reg, n_layers=L,
        hidden_dim=d, basename=basename, out_dir=out_dir)
    kwargs_list = [dict(seed=seed) for seed in SEEDS]
    results = run_parallel(func, kwargs_list, workers=multiprocess)
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f'{basename}.csv')
    print(f'Saving results to {csv_path}')
    df.to_csv(csv_path, index=False)

    # then optimize
    for alpha in alphas:
        print(f'Optimizing for alpha {alpha:.2f}')
        func = functools.partial(
            optimize_single, ds=ds, shuffle=shuffle, alpha=alpha, n_layers=L,
            hidden_dim=d, basename=basename, out_dir=out_dir)
        kwargs_list = [dict(seed=seed) for seed in SEEDS]
        results = run_parallel(func, kwargs_list, workers=multiprocess)

        df = pd.DataFrame(results)
        filename = f'eto_a{alpha:.2f}_L{L}_d{d}_lr{lr:.3g}_reg{l2reg:.3g}.csv'
        csv_path = os.path.join(out_dir, filename)
        print(f'Saving results to {csv_path}')
        df.to_csv(csv_path, index=False)


def e2e_single(
    seed: int, ds: str, alpha: float, lr: float, l2reg: float, n_layers: int,
    hidden_dim: int, large_q_penalty: float, shuffle: bool, basename: str, out_dir: str,
    saved_model_basename: str
) -> tuple[dict[str, Any], str]:
    loaders, (y_mean, y_std) = get_loaders(
        ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)
    prob = PortfolioProblemPICNN(
        N=Y_DIM[ds], L=n_layers, d=hidden_dim, y_mean=y_mean, y_std=y_std,
        epsilon=EPSILON)
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {'seed': seed}

    # load pre-trained PICNN
    if saved_model_basename != '':
        saved_model_path = os.path.join(out_dir, f'{saved_model_basename}_seed{seed}.pt')
    else:
        saved_model_path = ''

    model = PICNN(input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds], hidden_dim=hidden_dim,
                  n_layers=n_layers, y_in_output_layer=Y_IN_OUTPUT_LAYER, epsilon=EPSILON)
    model, train_result = train_e2e(
        model, loaders, alpha=alpha, max_epochs=MAX_E2E_EPOCHS, lr=lr, l2reg=l2reg,
        prob=prob, rng=rng, large_q_penalty=large_q_penalty,
        saved_model_path=saved_model_path)
    result |= train_result

    # evaluate on test
    opt_result = optimize_wrapper(model, prob=prob, loaders=loaders, alpha=alpha)
    result |= opt_result

    # save model
    ckpt_path = os.path.join(out_dir, f'{basename}_seed{seed}.pt')
    torch.save(model.cpu().state_dict(), ckpt_path)

    msg = (f'train e2e loss {result["train_task_losses"][-1]:.3f}, '
           f'val task loss {result["val_task_loss"]:.3f}, '
           f'best epoch {result["best_epoch"]}')
    return result, msg


def e2e(
    ds: str, alpha: float, lr: float, l2reg: float, n_layers: int, hidden_dim: int,
    large_q_penalty: float, shuffle: bool, out_dir: str, saved_lr: float | None,
    saved_l2reg: float | None, multiprocess: int = 1
) -> None:
    """
    Args:
        ds: dataset, either 'synthetic' or 'yfinance'
        alpha: risk level in (0, 1), the uncertainty set is a (1-alpha)-confidence set
        lr: learning rate
        l2reg: L2 regularization strength
        n_layers: # of hidden layers in PICNN
        hidden_dim: # of units per hidden layer
        large_q_penalty: weight for penalizing large q values
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        out_dir: where to save the results
        saved_lr: learning rate of the pre-trained model
        saved_l2reg: L2 regularization strength of the pre-trained model
        multiprocess: # of seeds to train in parallel
    """
    L = n_layers
    d = hidden_dim
    if saved_lr is not None:
        assert saved_l2reg is not None
        saved_model_basename = f'eto_L{L}_d{d}_lr{saved_lr:.3g}_reg{saved_l2reg:.3g}'
        tag = '_finetune'
    else:
        assert saved_l2reg is None
        saved_model_basename = ''
        tag = ''

    basename = f'e2e{tag}_a{alpha:.2f}_L{L}_d{d}_lr{lr:.3g}_reg{l2reg:.3g}'

    func = functools.partial(
        e2e_single, ds=ds, alpha=alpha, lr=lr, l2reg=l2reg, n_layers=L,
        hidden_dim=d, large_q_penalty=large_q_penalty, shuffle=shuffle,
        basename=basename, out_dir=out_dir, saved_model_basename=saved_model_basename)
    kwargs_list = [dict(seed=seed) for seed in SEEDS]
    results = run_parallel(func, kwargs_list, workers=multiprocess)
    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f'{basename}.csv')
    print(f'Saving results to {csv_path}')
    df.to_csv(csv_path, index=False)


def main(args: argparse.Namespace) -> None:
    tag = args.tag
    shuffle = args.shuffle
    ds = args.dataset

    if ds == 'synthetic':
        assert not args.shuffle
        out_dir = f'out/portfolio_syn_picnn{tag}/'
        SAVED_LR = 1e-2
        SAVED_L2REG = 1e-3
    elif ds == 'yfinance':
        if shuffle:
            out_dir = f'out/portfolio_yf_picnn{tag}_shuffle/'
            SAVED_LR = None
            SAVED_L2REG = None
        else:
            out_dir = f'out/portfolio_yf_picnn{tag}/'
            SAVED_LR = None
            SAVED_L2REG = None
    else:
        raise ValueError(f'Unknown dataset: {ds}')
    os.makedirs(out_dir, exist_ok=True)

    if args.command == 'eto':
        for lr, l2reg in itertools.product(args.lr, args.l2reg):
            print(f'ETO: lr={lr:.2g}, l2reg={l2reg:.3g}')
            eto(ds=ds, shuffle=shuffle, alphas=args.alpha, lr=lr, l2reg=l2reg,
                n_layers=N_LAYERS, hidden_dim=HIDDEN_DIM, out_dir=out_dir,
                multiprocess=args.multiprocess)

    elif args.command == 'e2e':
        for alpha, lr in itertools.product(args.alpha, args.lr):
            print(f'E2E: alpha={alpha:.2f}, lr={lr:.3g}')
            e2e(ds=ds, alpha=alpha, lr=lr, l2reg=SAVED_L2REG, n_layers=N_LAYERS,
                hidden_dim=HIDDEN_DIM, large_q_penalty=LARGE_Q_PENALTY, shuffle=shuffle,
                out_dir=out_dir, saved_lr=SAVED_LR, saved_l2reg=SAVED_L2REG,
                multiprocess=args.multiprocess)

    else:
        raise ValueError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    # For ETO, we search over lr in [1e-2, 1e-3, 1e-4] and l2reg in [1e-4, 1e-3, 1e-2]
    # For E2E, we use the L2REG found from ETO and tune lr in [1e-3, 1e-4]
    commands = ('eto', 'e2e')
    datasets = ('synthetic', 'yfinance')
    args = parse_args(commands, lr=True, l2reg=True, datasets=datasets)
    main(args)
