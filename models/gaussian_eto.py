from collections.abc import Iterable, Mapping

import numpy as np
import torch.utils.data
from tqdm.auto import tqdm

from problems.protocols import EllipsoidProblemProtocol
from .gaussian import GaussianRegressor, conformal_q, run_epoch_gaussian_regressor


def optimize(
    model: GaussianRegressor,
    prob: EllipsoidProblemProtocol,
    loader: torch.utils.data.DataLoader,
    q: float,
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
        pred = model(x_batch)
        loc_batch = pred[0].detach().cpu().numpy()
        scale_tril_batch = pred[1].detach().cpu().numpy()
        y_batch = y_batch.detach().cpu().numpy()

        # adjust scale_tril by conformal q
        scale_tril_batch *= np.sqrt(q)

        for y, loc, scale_tril in zip(y_batch, loc_batch, scale_tril_batch):
            prob.solve(loc, scale_tril)
            task_loss = prob.task_loss_np(y, is_standardized=True)
            task_losses.append(task_loss)
            if show_pbar:
                pbar.update(1)

    return np.mean(task_losses).item()


def optimize_wrapper(
    model: GaussianRegressor,
    prob: EllipsoidProblemProtocol,
    loaders: Mapping[str, torch.utils.data.DataLoader],
    y_dim: int,
    alpha: float,
    device: str = 'cpu',
    splits: Iterable[str] = ('train', 'test'),
) -> dict[str, float]:
    """Runs the "optimize" part of ETO on both train and test splits.

    Returns:
        result: dictionary of results
    """
    model.eval()
    with torch.no_grad():
        q = conformal_q(model, loaders['calib'], alpha=alpha).item()
        result = {'q': q}

        for split in splits:
            # coverage without conformal prediction
            nll_loss, coverage_no_conformal = run_epoch_gaussian_regressor(
                model, loaders[split], y_dim=y_dim, alpha=alpha, device=device)
            assert coverage_no_conformal is not None

            # coverage and task loss with conformal
            _, coverage = run_epoch_gaussian_regressor(
                model, loaders[split], y_dim=y_dim, q=q, device=device)
            assert coverage is not None
            task_loss = optimize(
                model, prob=prob, loader=loaders[split], q=q, device=device)

            if split == 'calib':
                split = 'val'
            result |= {
                f'{split}_coverage_no_conformal': coverage_no_conformal,
                f'{split}_coverage': coverage,
                f'{split}_nll_loss': nll_loss,
                f'{split}_task_loss': task_loss,
            }
    return result
