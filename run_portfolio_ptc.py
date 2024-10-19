"""Implements PTC baselines for portfolio optimization problem.

[1] Predict-then-Calibrate: A New Perspective of Robust Contextual LP
    Sun et al., 2023
    https://proceedings.neurips.cc/paper_files/paper/2023/hash/397271e11322fae8ba7f827c50ca8d9b-Abstract-Conference.html

[2] Conformal uncertainty sets for robust optimization
    Johnstone and Cox, 2021
    https://proceedings.mlr.press/v152/johnstone21a.html
"""
import argparse
import functools
import itertools
import os

import numpy as np
import pandas as pd
import torch.nn
import torch.utils.data
from tqdm.auto import tqdm

from models.ptc import (
    MLP, conformal_q_box, conformal_q_ellipse, conformal_q_ellipse_johnstone,
    get_residual_ds, optimize_wrapper_box, optimize_wrapper_ellipse,
    optimize_wrapper_ellipse_johnstone, train_mlp_regressor)
from models.quantile import PinballLoss
from portfolio import get_loaders
from portfolio.problems import PortfolioProblemBox, PortfolioProblemEllipsoid
from utils.data import split_loader
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
BATCH_SIZE = 256
MAX_EPOCHS = 100
SEEDS = range(10)


def ptc_box(
    ds: str, shuffle: bool, alpha: float, device: str, out_dir: str
) -> tuple[pd.DataFrame, str]:
    """Implements Predict-then-calibrate with box uncertainty.

    See Algorithm 1 from "Predict-then-Calibrate" (Sun et al., 2023)
    """
    pinball_loss_fn = PinballLoss(q=1-alpha)
    results = []
    for seed in tqdm(SEEDS):
        loaders, (y_mean, y_std) = get_loaders(
            ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)
        prob = PortfolioProblemBox(N=Y_DIM[ds], y_mean=y_mean, y_std=y_std)

        mlp = MLP(input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds])
        ckpt_path = os.path.join(out_dir, f'mlp_seed{seed}.pt')
        mlp.load_state_dict(torch.load(ckpt_path, weights_only=True))

        d1, d2 = split_loader(loaders['calib'])
        r1 = get_residual_ds(mlp, d1, transform='abs', device=device)
        r2 = get_residual_ds(mlp, d2, transform='abs', device=device)
        residual_loaders = dict(
            train=torch.utils.data.DataLoader(r1, batch_size=BATCH_SIZE, shuffle=True),
            calib=torch.utils.data.DataLoader(r2, batch_size=BATCH_SIZE, shuffle=False)
        )

        # train MLP to predict α-quantile of absolute residuals
        best_pinball_loss = np.inf
        best_quantile_mlp = None
        best_hps = (np.nan, np.nan)
        for lr, l2reg in itertools.product((1e-4, 1e-3, 1e-2), (0, 1e-4, 1e-3, 1e-2)):
            quantile_mlp, quantile_mlp_result = train_mlp_regressor(
                residual_loaders, pinball_loss_fn, input_dim=INPUT_DIM[ds],
                y_dim=Y_DIM[ds], max_epochs=MAX_EPOCHS, lr=lr, l2reg=l2reg,
                return_best_model=True, device=device, pos=True)
            if quantile_mlp_result['val_loss'] < best_pinball_loss:
                best_pinball_loss = quantile_mlp_result['val_loss']
                best_quantile_mlp = quantile_mlp
                best_hps = (lr, l2reg)
        tqdm.write(f'Best hyperparameters: lr={best_hps[0]}, l2reg={best_hps[1]}')
        assert best_quantile_mlp is not None
        quantile_mlp = best_quantile_mlp
        ckpt_path = os.path.join(out_dir, f'absres_quantile_mlp_a{alpha:.2f}_s{seed}.pt')
        torch.save(quantile_mlp.cpu().state_dict(), ckpt_path)
        tqdm.write(f'Saved quantile_mlp to {ckpt_path}')

        # then optimize
        q = conformal_q_box(mlp, quantile_mlp, d2, alpha=alpha, device=device).item()
        result = optimize_wrapper_box(
            mlp, quantile_mlp, prob=prob, loaders=loaders, q=q, device=device)
        result['seed'] = seed
        results.append(result)

        msg = (f'a{alpha:.2g}, s{seed} train task loss {result["train_task_loss"]:.3f}, '
               f'test task loss {result["test_task_loss"]:.3f}, '
               f'test coverage {result["test_coverage"]:.3f}')
        tqdm.write(msg)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f'ptc_box_a{alpha:.2f}.csv')
    df.to_csv(csv_path, index=False)
    return df, f'Saving results to {csv_path}'


def ptc_ellipse(
    ds: str, shuffle: bool, alpha: float, device: str, out_dir: str
) -> tuple[pd.DataFrame, str]:
    """Implements Predict-then-calibrate with ellipse uncertainty.

    See Algorithm 2 from "Predict-then-Calibrate" (Sun et al., 2023)
    """
    pinball_loss_fn = PinballLoss(q=1-alpha)
    results = []
    for seed in tqdm(SEEDS):
        loaders, (y_mean, y_std) = get_loaders(
            ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)
        prob = PortfolioProblemEllipsoid(N=Y_DIM[ds], y_mean=y_mean, y_std=y_std)

        mlp = MLP(input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds])
        ckpt_path = os.path.join(out_dir, f'mlp_seed{seed}.pt')
        mlp.load_state_dict(torch.load(ckpt_path, weights_only=True))

        d1, d2 = split_loader(loaders['calib'])
        ckpt_path = os.path.join(out_dir, f'normres_quantile_mlp_a{alpha:.2f}_s{seed}.pt')
        if os.path.exists(ckpt_path):
            tqdm.write(f'Found saved quantile_mlp at {ckpt_path}')
            quantile_mlp = MLP(input_dim=INPUT_DIM[ds], y_dim=1, pos=True)
            quantile_mlp.load_state_dict(torch.load(ckpt_path, weights_only=True))
        else:
            r1 = get_residual_ds(mlp, d1, transform='norm', device=device)
            r2 = get_residual_ds(mlp, d2, transform='norm', device=device)
            residual_loaders = dict(
                train=torch.utils.data.DataLoader(r1, batch_size=BATCH_SIZE, shuffle=True),
                calib=torch.utils.data.DataLoader(r2, batch_size=BATCH_SIZE, shuffle=False)
            )

            # train MLP to predict α-quantile of norm of residuals
            best_pinball_loss = np.inf
            best_quantile_mlp = None
            best_hps = (np.nan, np.nan)
            for lr, l2reg in itertools.product((1e-4, 1e-3, 1e-2), (0, 1e-4, 1e-3, 1e-2)):
                quantile_mlp, quantile_mlp_result = train_mlp_regressor(
                    residual_loaders, pinball_loss_fn, input_dim=INPUT_DIM[ds], y_dim=1,
                    max_epochs=MAX_EPOCHS, lr=lr, l2reg=l2reg, return_best_model=True,
                    device=device, pos=True)
                if quantile_mlp_result['val_loss'] < best_pinball_loss:
                    best_pinball_loss = quantile_mlp_result['val_loss']
                    best_quantile_mlp = quantile_mlp
                    best_hps = (lr, l2reg)
            tqdm.write(f'Best hyperparameters: lr={best_hps[0]}, l2reg={best_hps[1]}')
            assert best_quantile_mlp is not None
            quantile_mlp = best_quantile_mlp
            torch.save(quantile_mlp.cpu().state_dict(), ckpt_path)
            tqdm.write(f'Saved quantile_mlp to {ckpt_path}')

        # then optimize
        q, L = conformal_q_ellipse(mlp, quantile_mlp, d2, alpha=alpha, device=device)
        result = optimize_wrapper_ellipse(
            mlp, quantile_mlp, prob=prob, loaders=loaders, q=q.item(), L=L,
            device=device)
        result['seed'] = seed
        results.append(result)

        msg = (f'a{alpha:.2g}, s{seed} train task loss {result["train_task_loss"]:.3f}, '
               f'test task loss {result["test_task_loss"]:.3f}, '
               f'test coverage {result["test_coverage"]:.3f}')
        tqdm.write(msg)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f'ptc_ellipse_a{alpha:.2f}.csv')
    df.to_csv(csv_path, index=False)
    return df, f'Saving results to {csv_path}'


def ptc_ellipse_johnstone(
    ds: str, shuffle: bool, alpha: float, device: str, out_dir: str
) -> tuple[pd.DataFrame, str]:
    """Implements Predict-then-calibrate with ellipse uncertainty.

    Based on "Conformal uncertainty sets for robust optimization"
    (Johnstone and Cox, 2021).
    """
    results = []
    for seed in tqdm(SEEDS):
        loaders, (y_mean, y_std) = get_loaders(
            ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)
        prob = PortfolioProblemEllipsoid(N=Y_DIM[ds], y_mean=y_mean, y_std=y_std)

        # load point estimate model
        mlp = MLP(input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds])
        ckpt_path = os.path.join(out_dir, f'mlp_seed{seed}.pt')
        mlp.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # then optimize
        q, L = conformal_q_ellipse_johnstone(mlp, loaders, alpha=alpha, device=device)
        result = optimize_wrapper_ellipse_johnstone(
            mlp, prob=prob, loaders=loaders, q=q.item(), L=L, device=device)
        result['seed'] = seed
        results.append(result)

        msg = (f'a{alpha:.2g}, s{seed} train task loss {result["train_task_loss"]:.3f}, '
               f'test task loss {result["test_task_loss"]:.3f}, '
               f'test coverage {result["test_coverage"]:.3f}')
        tqdm.write(msg)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f'ptc_ellipse_johnstone_a{alpha:.2f}.csv')
    df.to_csv(csv_path, index=False)
    return df, f'Saving results to {csv_path}'


def get_best_hyperparams(ds: str, shuffle: bool, device: str, out_dir: str) -> None:
    """Finds best learning rate and L2 regularization strength for MLP regressor.

    Saves a CSV file to "out_dir/hyperparams.csv" with columns:
        lr, l2reg, seed, loss

    Args:
        ds: dataset, either 'synthetic' or 'yfinance'
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        device: 'cpu' or 'cuda'
        out_dir: where to save the results

    Returns:
        best_lr: best learning rate
        best_l2reg: best L2 regularization strength
    """
    lrs = 10. ** np.arange(-4.5, -1.4, 0.5)
    l2regs = [0, 1e-4, 1e-3, 1e-2]
    loss_fn = torch.nn.MSELoss()

    losses = []
    pbar = tqdm(total=len(SEEDS) * len(lrs) * len(l2regs))
    for seed in SEEDS:
        loaders, _ = get_loaders(ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)

        best_val_loss = np.inf
        best_model = None
        ckpt_path = os.path.join(out_dir, f'mlp_seed{seed}.pt')

        for lr, l2reg in itertools.product(lrs, l2regs):
            model, result = train_mlp_regressor(
                loaders, loss_fn, input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds],
                max_epochs=MAX_EPOCHS, lr=lr, l2reg=l2reg, device=device,
                return_best_model=True)
            losses.append((lr, l2reg, seed, result['val_loss']))
            pbar.update(1)

            if result['val_loss'] < best_val_loss:
                best_val_loss = result['val_loss']
                best_model = model

        assert best_model is not None
        torch.save(best_model.cpu().state_dict(), ckpt_path)

    df = pd.DataFrame(losses, columns=['lr', 'l2reg', 'seed', 'loss'])
    df.to_csv(os.path.join(out_dir, 'hyperparams.csv'), index=False)


def main(args: argparse.Namespace) -> None:
    tag = args.tag
    device = args.device
    shuffle = args.shuffle
    ds = args.dataset

    if ds == 'synthetic':
        assert not args.shuffle
        out_dir = f'out/portfolio_syn_ptc{tag}/'
    else:
        raise ValueError(f'Unknown dataset: {ds}')
    os.makedirs(out_dir, exist_ok=True)

    if args.command == 'best_hp':
        get_best_hyperparams(ds, shuffle=shuffle, device=device, out_dir=out_dir)

    elif args.command == 'ptc_box':
        func = functools.partial(
            ptc_box, ds=ds, shuffle=shuffle, device=device, out_dir=out_dir)
        kwargs_list = [dict(alpha=alpha) for alpha in args.alpha]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    elif args.command == 'ptc_ellipse':
        func = functools.partial(
            ptc_ellipse, ds=ds, shuffle=shuffle, device=device, out_dir=out_dir)
        kwargs_list = [dict(alpha=alpha) for alpha in args.alpha]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    elif args.command == 'ptc_ellipse_johnstone':
        func = functools.partial(
            ptc_ellipse_johnstone, ds=ds, shuffle=shuffle, device=device,
            out_dir=out_dir)
        kwargs_list = [dict(alpha=alpha) for alpha in args.alpha]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    else:
        raise ValueError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    commands = ('best_hp', 'ptc_box', 'ptc_ellipse', 'ptc_ellipse_johnstone')
    datasets = ('synthetic', 'yfinance')
    args = parse_args(commands, lr=False, l2reg=False, datasets=datasets)
    main(args)
