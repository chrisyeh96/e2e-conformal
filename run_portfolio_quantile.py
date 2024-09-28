import argparse
import functools
import itertools
import os
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from models.quantile import QuantileRegressor, train_quantile_regressor
from models.quantile_e2e import train_e2e
from models.quantile_eto import optimize_wrapper
from portfolio import get_loaders
from portfolio.problems import PortfolioProblemBox
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


def eto(
    ds: str, shuffle: bool, alpha: float, device: str, out_dir: str
) -> tuple[pd.DataFrame, str]:
    """
    Saves a CSV file with columns:
        seed, q, {train/val/test}_coverage_no_conformal,
        {train/val/test}_coverage_steps_no_conformal, {train/test}_coverage,
        {train/test}_coverage_steps, {train/val/test}_pinball_loss,
        {train/test}_task_loss

    Args:
        alpha: risk level in (0, 1), the uncertainty set is a (1-alpha)-confidence set
        lr: learning rate
        l2reg: L2 regularization strength
    """
    results = []
    for seed in tqdm(SEEDS):
        loaders, y_info = get_loaders(
            ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)
        y_mean, y_std = y_info
        prob = PortfolioProblemBox(N=Y_DIM[ds], y_mean=y_mean, y_std=y_std)
        result = {'seed': seed}

        # load pre-trained QR model
        ckpt_path = os.path.join(out_dir, f'quantile_regressor_a{alpha:.2f}_s{seed}.pt')
        model = QuantileRegressor(INPUT_DIM[ds], Y_DIM[ds])
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # then optimize
        opt_result = optimize_wrapper(
            model, prob=prob, loaders=loaders, y_dim=Y_DIM[ds], alpha=alpha,
            y_info=y_info, device=device)
        result |= opt_result
        results.append(result)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f'eto_a{alpha:.2f}.csv')
    df.to_csv(csv_path, index=False)
    return df, f'Saving results to {csv_path}'


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
        saved_model_fmt: format string for the saved model, takes `alpha` and `seed` as
            parameters, e.g., 'quantile_regressor_a{alpha:.2f}_s{seed}.pt'
    """
    tag = '' if saved_model_fmt == '' else '_finetune'
    basename = f'e2e{tag}_a{alpha:.2f}_lr{lr:.3g}_reg{l2reg:.3g}'

    results = []
    for seed in tqdm(SEEDS):
        loaders, y_info = get_loaders(
            ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)
        y_mean, y_std = y_info
        prob = PortfolioProblemBox(N=Y_DIM[ds], y_mean=y_mean, y_std=y_std)
        rng = np.random.default_rng(seed)
        result: dict[str, Any] = {'seed': seed}

        # load pre-trained QR model
        if saved_model_fmt != '':
            saved_model_path = os.path.join(
                out_dir, saved_model_fmt.format(alpha=alpha, seed=seed))
        else:
            saved_model_path = ''

        model, train_result = train_e2e(
            loaders, input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds], alpha=alpha,
            max_epochs=MAX_EPOCHS, lr=lr, l2reg=l2reg, y_info=y_info,
            prob=prob, rng=rng, quantile_loss_frac=0.1, saved_model_path=saved_model_path)
        result |= train_result

        # evaluate on test
        opt_result = optimize_wrapper(
            model, prob=prob, loaders=loaders, y_dim=Y_DIM[ds], alpha=alpha,
            y_info=y_info)
        result |= opt_result
        results.append(result)

        # save model
        ckpt_path = os.path.join(out_dir, f'{basename}_s{seed}.pt')
        torch.save(model.cpu().state_dict(), ckpt_path)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f'{basename}.csv')
    df.to_csv(csv_path, index=False)
    return df, f'Saving results to {csv_path}'


def get_best_hyperparams(
    ds: str, shuffle: bool, alpha: float, device: str, out_dir: str
) -> tuple[None, str]:
    """Finds best learning rate and L2 regularization strength for quantile regressor.

    Saves a CSV file to "out_dir/hyperparams_a{alpha:.2f}.csv" with columns:
        lr, l2reg, seed, loss

    Args:
        ds: dataset, either 'synthetic' or 'yfinance'
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        alpha: risk level in (0, 1), predicted quantiles are α/2 and 1-α/2
        device: 'cpu' or 'cuda'
        out_dir: where to save the results

    Returns:
        best_lr: best learning rate
        best_l2reg: best L2 regularization strength
    """
    lrs = 10. ** np.arange(-4.5, -1.4, 0.5)
    l2regs = [0, 1e-4, 1e-3, 1e-2]

    losses = []
    pbar = tqdm(total=len(SEEDS) * len(lrs) * len(l2regs))
    for seed in SEEDS:
        loaders, _ = get_loaders(ds, batch_size=BATCH_SIZE, seed=seed, shuffle=shuffle)

        best_val_loss = np.inf
        best_model = None
        ckpt_path = os.path.join(out_dir, f'quantile_regressor_a{alpha:.2f}_s{seed}.pt')

        for lr, l2reg in itertools.product(lrs, l2regs):
            model, result = train_quantile_regressor(
                loaders, input_dim=INPUT_DIM[ds], y_dim=Y_DIM[ds], alpha=alpha,
                max_epochs=MAX_EPOCHS, lr=lr, l2reg=l2reg, return_best_model=True,
                device=device)
            losses.append((lr, l2reg, seed, result['val_pinball_loss']))
            pbar.update(1)

            if result['val_pinball_loss'] < best_val_loss:
                best_val_loss = result['val_pinball_loss']
                best_model = model

        assert best_model is not None
        torch.save(best_model.cpu().state_dict(), ckpt_path)

    df = pd.DataFrame(losses, columns=['lr', 'l2reg', 'seed', 'loss'])
    csv_path = os.path.join(out_dir, f'hyperparams_a{alpha:.2g}.csv')
    df.to_csv(csv_path, index=False)
    return None, f'Saved results to {csv_path}'


def main(args: argparse.Namespace) -> None:
    tag = args.tag
    device = args.device
    shuffle = args.shuffle
    ds = args.dataset

    if ds == 'synthetic':
        assert not args.shuffle
        out_dir = f'out/portfolio_syn_quantile{tag}/'
        # best lr and l2reg found by get_best_hyperparams()
        SAVED_L2REG = {a: 1e-4 for a in (0.01, 0.05, 0.1, 0.2)}

    elif ds == 'yfinance':
        if shuffle:
            out_dir = f'out/portfolio_yf_quantile{tag}_shuffle/'
            # best lr and l2reg found by get_best_hyperparams()
            SAVED_L2REG = {}
        else:
            out_dir = f'out/portfolio_yf_quantile{tag}/'
            # best lr and l2reg found by get_best_hyperparams()
            SAVED_L2REG = {}
    else:
        raise ValueError(f'Unknown dataset: {ds}')
    os.makedirs(out_dir, exist_ok=True)

    if args.command == 'best_hp':
        func = functools.partial(
            get_best_hyperparams, ds=ds, shuffle=shuffle, device=device, out_dir=out_dir)
        kwargs_list = [dict(alpha=alpha) for alpha in args.alpha]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    elif args.command == 'eto':
        func = functools.partial(
            eto, ds=ds, shuffle=shuffle, device=device, out_dir=out_dir)
        kwargs_list = [dict(alpha=alpha) for alpha in args.alpha]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    elif args.command == 'e2e':
        func = functools.partial(
            e2e, ds=ds, shuffle=shuffle, out_dir=out_dir,
            saved_model_fmt='quantile_regressor_a{alpha:.2f}_s{seed}.pt')
        kwargs_list = [
            dict(alpha=alpha, lr=lr, l2reg=SAVED_L2REG[alpha])
            for alpha, lr in itertools.product(args.alpha, args.lr)
        ]
        run_parallel(func, kwargs_list, workers=args.multiprocess)

    else:
        raise ValueError(f'Unknown command: {args.command}')


if __name__ == '__main__':
    # For E2E, we use the L2REG found from get_best_hyperparams() for each alpha,
    #     and we try lr in [1e-2, 1e-3, 1e-4].
    commands = ('best_hp', 'eto', 'e2e')
    datasets = ('synthetic', 'yfinance')
    args = parse_args(commands, lr=True, l2reg=False, datasets=datasets)
    main(args)
