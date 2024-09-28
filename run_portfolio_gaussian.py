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

from models.gaussian import GaussianRegressor, train_gaussian_regressor
from models.gaussian_e2e import train_e2e
from models.gaussian_eto import optimize_wrapper
from portfolio import get_loaders
from portfolio.problems import PortfolioProblemEllipsoid
from utils.multiprocess import run_parallel
from utils.parse_args import parse_args


INPUT_DIM = {
    'synthetic': 2,
    'yfinance': 21
}
Y_DIM = {
    'synthetic': 2,
    'yfinance': 15
}
BATCH_SIZE = 256
MAX_EPOCHS = 100
SEEDS = range(10)


def optimize_single(
    seed: int, ds: str, alpha: float, shuffle: bool, device: str,
    out_dir: str,
) -> tuple[dict[str, float], str]:
    """Optimize over ellipsoid uncertainty set.

    Args:
        ds: dataset, either 'synthetic' or 'yfinance'
        device: either 'cpu' or 'cuda'

    Returns:
        result: dict of performance metrics
        msg: message string
    """
    loaders, (y_mean, y_std) = get_loaders(
        ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)
    prob = PortfolioProblemEllipsoid(N=Y_DIM[ds], y_mean=y_mean, y_std=y_std)

    # load estimated model
    ckpt_path = os.path.join(out_dir, f'gaussian_regressor_s{seed}.pt')
    model = GaussianRegressor(input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds])
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.to(device=device).eval()

    # then optimize
    result = optimize_wrapper(
        model, prob=prob, loaders=loaders, y_dim=Y_DIM[ds], alpha=alpha, device=device)
    result['seed'] = seed

    msg = (f'alpha {alpha:.2g} train task loss {result["train_task_loss"]:.3f}, '
           f'test task loss {result["test_task_loss"]:.3f}, '
           f'test coverage {result["test_coverage"]:.3f}')
    return result, msg


def eto(
    ds: str, shuffle: bool, alphas: Collection[float], device: str, out_dir: str,
    multiprocess: int = 1
) -> None:
    """
    Saves a CSV file with columns:
        seed, q, {train/val/test}_coverage_no_conformal,
        {train/val/test}_nll_loss, {train/test}_coverage,
        {train/test}_task_loss

    Args:
        ds: dataset, either 'synthetic' or 'yfinance'
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        alphas: risk levels in (0, 1), the uncertainty set is a (1-alpha)-confidence set
        device: either 'cpu' or 'cuda'
        out_dir: where to save the results
        multiprocess: number of processes to use, 1 = serial processing
    """
    for alpha in alphas:
        print(f'Optimizing for alpha {alpha:.2f}')
        func = functools.partial(
            optimize_single, ds=ds, shuffle=shuffle, alpha=alpha, device=device,
            out_dir=out_dir)
        kwargs_list = [dict(seed=seed) for seed in SEEDS]
        results = run_parallel(func, kwargs_list, workers=multiprocess)

        df = pd.DataFrame(results)
        csv_path = os.path.join(out_dir, f'eto_a{alpha:.2f}.csv')
        print(f'Saving results to {csv_path}')
        df.to_csv(csv_path, index=False)


def e2e(
    ds: str, shuffle: bool, alpha: float, lr: float, l2reg: float, out_dir: str,
    saved_model_fmt: str = ''
) -> tuple[pd.DataFrame, str]:
    """
    Everything in this function is done on CPU.

    Args:
        ds: dataset, either 'synthetic' or 'yfinance'
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        alpha: risk level in (0, 1), the uncertainty set is a (1-alpha)-confidence set
        lr: learning rate
        l2reg: L2 regularization strength
        out_dir: where to save the results
        saved_model_fmt: format string for the saved model, takes `seed` as parameter,
            e.g., 'gaussian_regressor_s{seed}.pt'
    """
    tag = '' if saved_model_fmt == '' else '_finetune'
    basename = f'e2e{tag}_a{alpha:.2f}_lr{lr:.3g}_reg{l2reg:.3g}'

    results = []
    for seed in tqdm(SEEDS):
        loaders, (y_mean, y_std) = get_loaders(ds, BATCH_SIZE, seed, shuffle)
        prob = PortfolioProblemEllipsoid(N=Y_DIM[ds], y_mean=y_mean, y_std=y_std)
        rng = np.random.default_rng(seed)
        result: dict[str, Any] = {'seed': seed}

        # load pre-trained GaussianRegressor
        if saved_model_fmt != '':
            saved_model_path = os.path.join(out_dir, saved_model_fmt.format(seed=seed))
        else:
            saved_model_path = ''

        model = GaussianRegressor(input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds])
        train_result = train_e2e(
            model, loaders, alpha=alpha, max_epochs=MAX_EPOCHS, lr=lr, l2reg=l2reg,
            prob=prob, rng=rng, nll_loss_frac=0.1, saved_model_path=saved_model_path)
        result |= train_result

        # evaluate on test
        opt_result = optimize_wrapper(
            model, prob=prob, loaders=loaders, y_dim=Y_DIM[ds], alpha=alpha)
        result |= opt_result
        results.append(result)

        # save model
        ckpt_path = os.path.join(out_dir, f'{basename}_s{seed}.pt')
        torch.save(model.cpu().state_dict(), ckpt_path)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f'{basename}.csv')
    df.to_csv(csv_path, index=False)
    return df, f'Saving results to {csv_path}'


def get_best_hyperparams(ds: str, shuffle: bool, device: str, out_dir: str) -> None:
    """Finds best learning rate and L2 regularization strength for Gaussian regressor.

    Saves a CSV file to "out_dir/hyperparams.csv" with columns:
        lr, l2reg, seed, loss

    Args:
        ds: dataset, either 'synthetic' or 'yfinance'
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        device: 'cpu' or 'cuda'
        out_dir: where to save the results
    """
    lrs = 10. ** np.arange(-4.5, -1.4, 0.5)
    l2regs = [0, 1e-4, 1e-3, 1e-2]

    losses = []
    pbar = tqdm(total=len(SEEDS) * len(lrs) * len(l2regs))
    for seed in SEEDS:
        loaders, _ = get_loaders(ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)

        best_val_loss = np.inf
        best_model = None
        ckpt_path = os.path.join(out_dir, f'gaussian_regressor_s{seed}.pt')

        for lr, l2reg in itertools.product(lrs, l2regs):
            try:
                model, result = train_gaussian_regressor(
                    loaders, input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds],
                    max_epochs=MAX_EPOCHS, lr=lr, l2reg=l2reg, return_best_model=True,
                    device=device)
                losses.append((lr, l2reg, seed, result['val_nll_loss']))

                if result['val_nll_loss'] < best_val_loss:
                    best_val_loss = result['val_nll_loss']
                    best_model = model
            except Exception as e:
                tqdm.write(f'(lr {lr:.3g}, l2reg {l2reg:.3g}, seed {seed}) failed: {e}')
                losses.append((lr, l2reg, seed, np.nan))
            pbar.update(1)

        assert best_model is not None
        torch.save(best_model.cpu().state_dict(), ckpt_path)

    df = pd.DataFrame(losses, columns=['lr', 'l2reg', 'seed', 'loss'])
    csv_path = os.path.join(out_dir, 'hyperparams.csv')
    df.to_csv(csv_path, index=False)
    print(f'Saved results to {csv_path}')


def main(args: argparse.Namespace) -> None:
    tag = args.tag
    device = args.device
    shuffle = args.shuffle
    ds = args.dataset

    if ds == 'synthetic':
        assert not args.shuffle
        out_dir = f'out/portfolio_syn_gaussian{tag}/'
        # best lr and l2reg found by get_best_hyperparams()
        L2REG = 1e-2

    elif ds == 'yfinance':
        if shuffle:
            out_dir = f'out/portfolio_yf_gaussian{tag}_shuffle/'
            # best lr and l2reg found by get_best_hyperparams()
            L2REG = None
        else:
            out_dir = f'out/portfolio_yf_gaussian{tag}/'
            # best lr and l2reg found by get_best_hyperparams()
            L2REG = None
    else:
        raise ValueError(f'Unknown dataset: {ds}')
    os.makedirs(out_dir, exist_ok=True)

    if args.command == 'best_hp':
        get_best_hyperparams(ds, shuffle=shuffle, device=device, out_dir=out_dir)

    elif args.command == 'eto':
        eto(ds=ds, shuffle=shuffle, alphas=args.alpha, device=device, out_dir=out_dir,
            multiprocess=args.multiprocess)

    elif args.command == 'e2e':
        func = functools.partial(
            e2e, ds=ds, shuffle=shuffle, l2reg=L2REG, out_dir=out_dir,
            saved_model_fmt='gaussian_regressor_s{seed}.pt')
        kwargs_list = [
            dict(alpha=alpha, lr=lr)
            for alpha, lr in itertools.product(args.alpha, args.lr)
        ]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    else:
        raise ValueError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    # For E2E, we use the L2REG found from get_best_hyperparams(), and tune
    #     lr in [1e-2, 1e-3, 1e-4].
    commands = ('best_hp', 'eto', 'e2e')
    datasets = ('synthetic', 'yfinance')
    args = parse_args(commands, lr=True, l2reg=False, datasets=datasets)
    main(args)
