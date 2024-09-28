from collections.abc import Mapping

import numpy as np
import torch.utils.data
from tqdm.auto import tqdm

from problems.protocols import BoxProblemProtocol
from .quantile import QuantileRegressor, conformal_q, run_epoch_quantile_regressor


def optimize(
    model: QuantileRegressor,
    prob: BoxProblemProtocol,
    loader: torch.utils.data.DataLoader,
    y_dim: int,
    q: float,
    y_info: str | tuple[np.ndarray, np.ndarray],
    device: str = 'cpu',
    show_pbar: bool = False
) -> float:
    """
    Args:
        model: model to evaluate
        prob: robust optimization problem to solve (with box constraints)
        loader: data loader
        y_dim: dimension of each label
        q: score threshold for the desired coverage level
        y_info: either string 'log' or a tuple (mean, std) of the label
        device: device to use
        show_pbar: whether to show a progress bar

    Returns:
        task_loss: average task loss per training example
    """
    model.eval()
    model = model.to(device)

    task_losses = []
    if show_pbar:
        pbar = tqdm(total=len(loader.dataset))
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)

        pred = model(x_batch).detach().cpu().numpy()
        y_batch = y_batch.detach().cpu().numpy()

        # adjust lo and hi preds by conformal q
        pred_lo_batch = pred[..., :y_dim] - q
        pred_hi_batch = pred[..., y_dim:] + q

        if y_info == 'log':
            y_batch = np.exp(y_batch)
            pred_lo_batch = np.exp(pred_lo_batch)
            pred_hi_batch = np.exp(pred_hi_batch)
            is_standardized = False
        else:
            is_standardized = True

        for y, pred_lo, pred_hi in zip(y_batch, pred_lo_batch, pred_hi_batch):
            prob.solve(pred_lo, pred_hi)
            task_loss = prob.task_loss_np(y, is_standardized=is_standardized)
            task_losses.append(task_loss)
            if show_pbar:
                pbar.update(1)

    return np.mean(task_losses).item()


def optimize_wrapper(
    model: QuantileRegressor,
    prob: BoxProblemProtocol,
    loaders: Mapping[str, torch.utils.data.DataLoader],
    y_dim: int,
    alpha: float,
    y_info: str | tuple[np.ndarray, np.ndarray],
    device: str = 'cpu'
) -> dict[str, float]:
    """Runs the "optimize" part of ETO on both train and test splits.

    Returns:
        result: dictionary of results, with keys for split in ['train', 'test']
            'q', '{split}_coverage_no_conformal', '{split}_coverage_steps_no_conformal',
            '{split}_coverage', '{split}_coverage_steps', '{split}_pinball_loss',
            '{split}_task_loss'
    """
    model.eval()
    with torch.no_grad():
        q = conformal_q(model, loaders['calib'], y_dim=y_dim, alpha=alpha).item()
        result = {'q': q}

        for split in ('train', 'test'):
            # coverage without conformal prediction
            coverage_no_conformal, coverage_steps_no_conformal, pinball_loss = run_epoch_quantile_regressor(
                model, loaders[split], y_dim=y_dim, alpha=alpha, q=0, device=device)

            # coverage, pinball loss, and task loss with conformal
            coverage, coverage_steps, _ = run_epoch_quantile_regressor(
                model, loaders[split], y_dim=y_dim, alpha=alpha, q=q)
            task_loss = optimize(
                model, prob=prob, loader=loaders[split], y_dim=y_dim, q=q, y_info=y_info,
                device=device)

            result |= {
                f'{split}_coverage_no_conformal': coverage_no_conformal,
                f'{split}_coverage_steps_no_conformal': coverage_steps_no_conformal,
                f'{split}_coverage': coverage,
                f'{split}_coverage_steps': coverage_steps,
                f'{split}_pinball_loss': pinball_loss,
                f'{split}_task_loss': task_loss,
            }
    return result
