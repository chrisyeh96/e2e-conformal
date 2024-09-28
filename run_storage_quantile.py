import argparse
import functools
import itertools
import os

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from models.quantile import QuantileRegressor, train_quantile_regressor
from models.quantile_e2e import train_e2e
from models.quantile_eto import optimize_wrapper
from storage.data import get_loaders, get_tensors, get_train_calib_split
from storage.problems import StorageProblemBoxV2
from utils.parse_args import parse_args
from utils.multiprocess import run_parallel


INPUT_DIM = 101
Y_DIM = 24
BATCH_SIZE = 256
MAX_EPOCHS = 100
SEEDS = range(10)

# if True, predict quantiles of log energy prices; otherwise, predict quantiles of energy prices
LOG_PRICES = False


def eto(
    shuffle: bool, alpha: float, device: str, out_dir: str
) -> tuple[pd.DataFrame, str]:
    """
    Saves a CSV file with columns:
        seed, q, {train/val/test}_coverage_no_conformal,
        {train/val/test}_coverage_steps_no_conformal, {train/test}_coverage,
        {train/test}_coverage_steps, {train/val}_pinball_loss, {train/test}_task_loss

    Args:
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        alpha: risk level in (0, 1), the uncertainty set is a (1-alpha)-confidence set
        device: either 'cpu' or 'cuda'
        out_dir: where to save the results
    """
    tensors, y_info = get_tensors(shuffle=shuffle, log_prices=LOG_PRICES)
    if isinstance(y_info, str):
        assert y_info == 'log'
        prob = StorageProblemBoxV2(T=Y_DIM, y_mean=np.array(0.), y_std=np.array(1.))
    else:
        y_mean, y_std = y_info
        prob = StorageProblemBoxV2(T=Y_DIM, y_mean=y_mean, y_std=y_std)

    results = []
    pbar = tqdm(SEEDS)
    for seed in pbar:
        tensors_cv, _ = get_train_calib_split(tensors, seed=seed)
        loaders = get_loaders(tensors_cv, batch_size=BATCH_SIZE)
        result = {'seed': seed}

        # load pre-trained QR model
        ckpt_path = os.path.join(out_dir, f'quantile_regressor_a{alpha:.2f}_s{seed}.pt')
        model = QuantileRegressor(INPUT_DIM, Y_DIM)
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        # then optimize
        opt_result = optimize_wrapper(
            model, prob=prob, loaders=loaders, y_dim=Y_DIM, alpha=alpha, y_info=y_info,
            device=device)
        result |= opt_result
        results.append(result)

        msg = (f'alpha {alpha:.2g} train task loss {result["train_task_loss"]:.3f}, '
               f'test task loss {result["test_task_loss"]:.3f}, '
               f'test coverage {result["test_coverage"]:.3f}')
        pbar.set_description(msg)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f'eto_a{alpha:.2f}.csv')
    df.to_csv(csv_path, index=False)
    return df, f'Saving results to {csv_path}'


def e2e(
    shuffle: bool, alpha: float, lr: float, l2reg: float, out_dir: str,
    saved_model_fmt: str = ''
) -> tuple[pd.DataFrame, str]:
    """
    Everything in this function is done on CPU.

    Args:
        alpha: risk level in (0, 1), the uncertainty set is a (1-alpha)-confidence set
        lr: learning rate
        l2reg: L2 regularization strength
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        out_dir: where to save the results
        saved_model_fmt: format string for the saved model, takes `alpha` and `seed` as
            parameters, e.g., 'quantile_regressor_a{alpha:.2f}_s{seed}.pt'
    """
    tensors, y_info = get_tensors(shuffle=shuffle, log_prices=LOG_PRICES)
    if isinstance(y_info, str):
        assert y_info == 'log'
        prob = StorageProblemBoxV2(T=Y_DIM, y_mean=np.array(0.), y_std=np.array(1.))
    else:
        y_mean, y_std = y_info
        prob = StorageProblemBoxV2(T=Y_DIM, y_mean=y_mean, y_std=y_std)

    tag = '' if saved_model_fmt == '' else '_finetune'
    basename = f'e2e{tag}_a{alpha:.2f}_lr{lr:.3g}_reg{l2reg:.3g}'

    results = []
    for seed in tqdm(SEEDS):
        tensors_cv, _ = get_train_calib_split(tensors, seed=seed)
        loaders = get_loaders(tensors_cv, batch_size=BATCH_SIZE)
        rng = np.random.default_rng(seed)
        result = {'seed': seed}

        # load pre-trained QR model
        if saved_model_fmt != '':
            saved_model_path = os.path.join(
                out_dir, saved_model_fmt.format(alpha=alpha, seed=seed))
        else:
            saved_model_path = ''

        model, train_result = train_e2e(
            loaders, input_dim=INPUT_DIM, y_dim=Y_DIM, alpha=alpha,
            max_epochs=MAX_EPOCHS, lr=lr, l2reg=l2reg, y_info=y_info,
            prob=prob, rng=rng, quantile_loss_frac=0.1, saved_model_path=saved_model_path)
        result |= train_result

        # evaluate on test
        opt_result = optimize_wrapper(
            model, prob=prob, loaders=loaders, y_dim=Y_DIM, alpha=alpha, y_info=y_info)
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
    shuffle: bool, alpha: float, device: str, out_dir: str
) -> tuple[None, str]:
    """Finds best learning rate and L2 regularization strength for quantile regressor.

    Args:
        shuffle: whether to shuffle dataset before splitting into train/calib/test
        alpha: risk level in (0, 1), predicted quantiles are α/2 and 1-α/2
        device: 'cpu' or 'cuda'
        out_dir: where to save the results
    """
    tensors, _ = get_tensors(shuffle=shuffle, log_prices=LOG_PRICES)

    lrs = 10. ** np.arange(-4.5, -1.4, 0.5)
    l2regs = [0, 1e-4, 1e-3, 1e-2]

    losses = []
    pbar = tqdm(total=len(SEEDS) * len(lrs) * len(l2regs))
    for seed in SEEDS:
        tensors_cv, _ = get_train_calib_split(tensors, seed=seed)
        loaders = get_loaders(tensors_cv, batch_size=BATCH_SIZE)

        best_val_loss = np.inf
        best_model = None
        ckpt_path = os.path.join(out_dir, f'quantile_regressor_a{alpha:.2f}_s{seed}.pt')

        for lr, l2reg in itertools.product(lrs, l2regs):
            model, result = train_quantile_regressor(
                loaders, input_dim=INPUT_DIM, y_dim=Y_DIM, alpha=alpha,
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

    if shuffle:
        out_dir = f'out/storage_quantile{tag}_shuffle/'
        # best lr and l2reg found by get_best_hyperparams()
        SAVED_L2REG = {0.01: 0, 0.05: 1e-4, 0.1: 1e-4, 0.2: 0}
    else:
        out_dir = f'out/storage_quantile{tag}/'
        # best lr and l2reg found by get_best_hyperparams()
        SAVED_L2REG = {0.01: 1e-4, 0.05: 1e-4, 0.1: 0, 0.2: 1e-3}
    os.makedirs(out_dir, exist_ok=True)

    if args.command == 'best_hp':
        func = functools.partial(
            get_best_hyperparams, device=device, shuffle=shuffle, out_dir=out_dir)
        kwargs_list = [dict(alpha=alpha) for alpha in args.alpha]
        run_parallel(func, kwargs_list, args.multiprocess)

    elif args.command == 'eto':
        func = functools.partial(eto, shuffle=shuffle, device=device, out_dir=out_dir)
        kwargs_list = [dict(alpha=alpha) for alpha in args.alpha]
        run_parallel(func, kwargs_list, args.multiprocess)

    elif args.command == 'e2e':
        func = functools.partial(
            e2e, shuffle=shuffle, out_dir=out_dir,
            saved_model_fmt='quantile_regressor_a{alpha:.2f}_s{seed}.pt')
        kwargs_list = [
            dict(alpha=alpha, lr=lr, l2reg=SAVED_L2REG[alpha])
            for alpha, lr in itertools.product(args.alpha, args.lr)
        ]
        run_parallel(func, kwargs_list, args.multiprocess)


if __name__ == '__main__':
    # For ETO, we use the best LR and L2REG found by get_best_hyperparams().
    # For E2E, we use the L2REG found from get_best_hyperparams(), and tune
    #     lr in [1e-2, 1e-3, 1e-4].
    commands = ('best_hp', 'eto', 'e2e')
    args = parse_args(commands, lr=True, l2reg=False)
    main(args)
