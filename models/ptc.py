from collections.abc import Mapping
import io
from typing import Any

import numpy as np
import torch.utils.data
from torch import nn, Tensor
import torch.nn.functional as F
from tqdm.auto import tqdm

from models.gaussian import mahalanobis_dist2
from problems.protocols import BoxProblemProtocol, EllipsoidProblemProtocol
from utils.conformal import calc_q


class MLP(nn.Module):
    """3-hidden-layer MLP with BatchNorm and ReLU activations.

    Args:
        input_dim: dimension of each input example x
        y_dim: dimension of each label y
        pos: whether to enforce positive outputs via softplus
    """
    def __init__(self, input_dim: int, y_dim: int, pos: bool = False):
        super().__init__()
        self.y_dim = y_dim
        self.pos = pos
        ReLU = nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            ReLU,
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            ReLU,
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            ReLU,
            nn.Linear(256, y_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.net(x)
        if self.pos:
            out = F.softplus(out) + 1e-6
        return out


def get_residual_ds(
    model: nn.Module,
    ds: torch.utils.data.TensorDataset,
    transform: str | None = None,
    device: str = 'cpu'
) -> torch.utils.data.TensorDataset:
    """
    Args:
        model: model that outputs a point estimate
        ds: dataset, with tensors (xs, ys)
            xs: shape [n, input_dim]
            ys: shape [n, y_dim]
        transform: optional, either 'abs' (elementwise) or 'norm'
            None: residuals are (ys - model(xs)), shape [n, y_dim]
            'abs': residuals are |ys - model(xs)|, shape [n, y_dim]
            'norm': residuals are ‖ys - model(xs)‖₂, shape [n, 1]
        device: device to use

    Returns:
        residuals_ds: dataset with tensors (xs, residuals)
    """
    model.eval().to(device)
    xs, ys = ds.tensors
    with torch.no_grad():
        residuals = ys.to(device) - model(xs.to(device))
        if transform == 'abs':
            residuals = torch.abs(residuals)
        elif transform == 'norm':
            residuals = torch.linalg.norm(residuals, dim=1, keepdim=True)
    return torch.utils.data.TensorDataset(xs, residuals.cpu())


def run_epoch_mlp(
    model: MLP,
    loss_fn: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = 'cpu'
) -> float:
    """Runs 1 epoch of training or evaluation with a MLP regression model.

    Args:
        model: model to train
        loader: data loader
        optimizer: optimizer to use for training, None for evaluation
        device: either 'cpu' or 'cuda'

    Returns:
        avg_loss: average squared error loss per training example
    """
    if optimizer is None:
        model.eval().to(device)
    else:
        model.train().to(device)

    total_loss = 0.
    num_total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        total_loss += loss.item() * y.shape[0]
        num_total += y.shape[0]

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss / num_total


def train_mlp_regressor(
    loaders: Mapping[str, torch.utils.data.DataLoader],
    loss_fn: nn.Module,
    input_dim: int,
    y_dim: int,
    max_epochs: int,
    lr: float,
    l2reg: float,
    show_pbar: bool = False,
    return_best_model: bool = False,
    device: str = 'cpu',
    pos: bool = False
) -> tuple[MLP, dict[str, Any]]:
    """Trains a MLP regression model.

    Uses early-stopping based on MSE loss on the calibration set.

    Args:
        loaders: dictionary of data loaders
        loss_fn: loss function
        input_dim: dimension of each input example x
        y_dim: dimension of each label y
        max_epochs: maximum number of epochs to train
        lr: learning rate
        l2reg: L2 regularization strength
        show_pbar: if True, show a progress bar
        return_best_model: if True, return the model with the best validation loss,
            otherwise returns the model from the last training epoch
        device: either 'cpu' or 'cuda'
        pos: whether to enforce positive outputs via softplus

    Returns:
        model: trained model on CPU, either the model from the last training epoch, or
            the model that achieves best result['val_loss'] on the calibration set,
            see `return_best_model`
        result: dict of performance metrics
    """
    # Initialize model
    model = MLP(input_dim=input_dim, y_dim=y_dim, pos=pos).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2reg)

    result: dict[str, Any] = {
        'train_losses': [],
        'val_losses': [],
        'best_epoch': 0,
        'val_loss': np.inf,  # best loss on val set
    }
    steps_since_decrease = 0
    buffer = io.BytesIO()

    for epoch in tqdm(range(max_epochs)) if show_pbar else range(max_epochs):
        # train
        train_loss = run_epoch_mlp(
            model, loss_fn, loaders['train'], optimizer=optimizer, device=device)
        result['train_losses'].append(train_loss)

        # calculate loss on calibration set
        with torch.no_grad():
            val_loss = run_epoch_mlp(model, loss_fn, loaders['calib'], device=device)
            result['val_losses'].append(val_loss)

        if show_pbar:
            tqdm.write(f'Train loss: {train_loss:.3f}, '
                       f'Calib loss: {val_loss:.3f}')

        steps_since_decrease += 1

        if val_loss < result['val_loss']:
            result['best_epoch'] = epoch
            result['val_loss'] = val_loss
            steps_since_decrease = 0
            if return_best_model:
                buffer.seek(0)
                torch.save(model.state_dict(), buffer)

        if steps_since_decrease > 10:
            break

    # load best model
    if return_best_model:
        buffer.seek(0)
        model.load_state_dict(torch.load(buffer, weights_only=True))

    return model.cpu(), result


def conformal_q_box(
    mlp: MLP,
    quantile_mlp: MLP,
    ds: torch.utils.data.TensorDataset,
    alpha: float,
    device: str = 'cpu'
) -> Tensor:
    """Computes score threshold.

    score(x, y) = max_i( |y-mlp(x)|_i / quantile_mlp(x)_i )

    Args:
        mlp: outputs point estimate
        quantile_mlp: estimates quantile of absolute residual vector
        ds: dataset
        alpha: desired coverage level in (0, 1)
        device: device to use

    Returns:
        q: scalar score threshold for the desired coverage level
    """
    assert quantile_mlp.pos

    mlp.eval().to(device)
    quantile_mlp.eval().to(device)
    xs = ds.tensors[0].to(device)
    ys = ds.tensors[1].to(device)

    with torch.no_grad():
        zs = torch.abs(ys - mlp(xs)) / quantile_mlp(xs)  # shape [n, y_dim]
        scores = torch.max(zs, dim=1)[0]

    q = calc_q(scores, alpha)
    assert q != torch.inf, 'Size of calibration set is too small'
    return q


def optimize_box(
    mlp: MLP,
    quantile_mlp: MLP,
    prob: BoxProblemProtocol,
    loader: torch.utils.data.DataLoader,
    q: float,
    device: str = 'cpu',
    show_pbar: bool = False
) -> tuple[float, float]:
    """
    This function should be run inside a `with torch.no_grad()` block.

    Args:
        mlp: already with .eval() and on device
        quantile_mlp: already with .eval() and on device
        prob: robust optimization problem to solve (with box constraints)
        loader: data loader
        q: score threshold for the desired coverage level
        device: device to use
        show_pbar: whether to show a progress bar

    Returns:
        avg_task_loss: average task loss per training example
        coverage: proportion of covered examples
    """
    if show_pbar:
        pbar = tqdm(total=len(loader.dataset))

    task_losses = []
    num_total = 0
    num_covered = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)

        pred = mlp(x_batch).cpu().numpy()
        pred_quantile = quantile_mlp(x_batch).cpu().numpy()
        y_batch = y_batch.cpu().numpy()

        # adjust lo and hi preds by conformal q
        spread = q * pred_quantile
        pred_lo_batch = pred - spread
        pred_hi_batch = pred + spread

        covered = (pred_lo_batch <= y_batch) & (y_batch <= pred_hi_batch)  # shape [batch, y_dim]
        num_covered += covered.all(axis=1).sum().item()
        num_total += y_batch.shape[0]

        for y, pred_lo, pred_hi in zip(y_batch, pred_lo_batch, pred_hi_batch):
            prob.solve(pred_lo, pred_hi)
            task_loss = prob.task_loss_np(y, is_standardized=True)
            task_losses.append(task_loss)
            if show_pbar:
                pbar.update(1)

    avg_task_loss = np.mean(task_losses).item()
    coverage = num_covered / num_total
    return avg_task_loss, coverage


def optimize_wrapper_box(
    mlp: MLP,
    quantile_mlp: MLP,
    prob: BoxProblemProtocol,
    loaders: Mapping[str, torch.utils.data.DataLoader],
    q: float,
    device: str = 'cpu',
) -> dict[str, float]:
    mlp.eval().to(device)
    quantile_mlp.eval().to(device)
    result = {'q': q}
    with torch.no_grad():
        for split in ('train', 'test'):
            task_loss, coverage = optimize_box(
                mlp, quantile_mlp, prob=prob, loader=loaders[split], q=q, device=device)
            result |= {
                f'{split}_task_loss': task_loss,
                f'{split}_coverage': coverage,
            }
    return result


def conformal_q_ellipse(
    mlp: MLP,
    quantile_mlp: MLP,
    ds: torch.utils.data.TensorDataset,
    alpha: float,
    device: str = 'cpu'
) -> tuple[Tensor, Tensor]:
    """Calculates the score threshold and covariance matrix for the scaled residuals.

    See Algorithm 2 from
        Predict-then-Calibrate: A New Perspective of Robust Contextual LP
        Sun et al., 2023
        https://proceedings.neurips.cc/paper_files/paper/2023/hash/397271e11322fae8ba7f827c50ca8d9b-Abstract-Conference.html

    Also see LUQ/train_2norm_h.py from the Sun et al. code repository (see their
    Supplemental zip file for their code). See in particular lines 300-302.

    Args:
        mlp: outputs point estimate
        quantile_mlp: estimates quantile of absolute residual vector
        ds: dataset
        alpha: desired coverage level in (0, 1)
        device: device to use

    Returns:
        q: scalar score threshold for the desired coverage level
        L: Cholesky factor of covariance matrix for scaled residuals
    """
    assert quantile_mlp.pos

    mlp.eval().to(device)
    quantile_mlp.eval().to(device)
    xs = ds.tensors[0].to(device)
    ys = ds.tensors[1].to(device)

    with torch.no_grad():
        pred = mlp(xs)  # shape [n, y_dim]
        rnorm_quantile = quantile_mlp(xs)  # shape [n, 1]
        R = (ys - pred) / rnorm_quantile  # shape [n, y_dim]

        # let S be the best-fit covariance matrix of a zero-mean
        # multivariate Gaussian for the scaled residuals
        S = R.mT @ R / (R.shape[0] - 1)  # shape [y_dim, y_dim]
        L = torch.linalg.cholesky(S)

        # if the scaled residuals were already zero-mean, then
        # we could directly calculate the sample covariance matrix
        # S = torch.cov(r_scaled.T)  # shape [y_dim, y_dim]
        # L = torch.linalg.cholesky()

        # calculate score
        scale_trils = rnorm_quantile[:, :, None] * L  # shape [n, y_dim, y_dim]
        scores = mahalanobis_dist2(loc=pred, scale_tril=scale_trils, y=ys)

    q = calc_q(scores, alpha)
    assert q != torch.inf, 'Size of calibration set is too small'
    return q, L


def optimize_ellipse(
    mlp: MLP,
    quantile_mlp: MLP,
    prob: EllipsoidProblemProtocol,
    loader: torch.utils.data.DataLoader,
    q: float,
    L: Tensor,
    device: str = 'cpu',
    show_pbar: bool = False
) -> tuple[float, float]:
    """
    This function should be run inside a `with torch.no_grad()` block.

    Args:
        mlp: already with .eval() and on device
        quantile_mlp: already with .eval() and on device
        prob: robust optimization problem to solve (with box constraints)
        loader: data loader
        q: score threshold for the desired coverage level
        L: Cholesky factor of covariance matrix for scaled residuals
        device: device to use
        show_pbar: whether to show a progress bar

    Returns:
        avg_task_loss: average task loss per training example
        coverage: proportion of covered examples
    """
    if show_pbar:
        pbar = tqdm(total=len(loader.dataset))

    task_losses = []
    num_total = 0
    num_covered = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        pred = mlp(x_batch)  # shape [n, y_dim]
        rnorm_quantile = quantile_mlp(x_batch)  # shape [n, 1]

        # calculate coverage
        scale_tril_batch = rnorm_quantile[:, :, None] * L  # shape [n, y_dim, y_dim]
        scores = mahalanobis_dist2(pred, scale_tril_batch, y_batch)
        covered = (scores <= q)  # shape [n]
        num_covered += int(covered.sum().item())
        num_total += y_batch.shape[0]

        # convert all to numpy
        scale_tril_batch = scale_tril_batch.cpu().numpy() * np.sqrt(q)
        loc_batch = pred.cpu().numpy()
        y_batch = y_batch.cpu().numpy()

        for y, loc, scale_tril in zip(y_batch, loc_batch, scale_tril_batch):
            prob.solve(loc, scale_tril)
            task_loss = prob.task_loss_np(y, is_standardized=True)
            task_losses.append(task_loss)
            if show_pbar:
                pbar.update(1)

    avg_task_loss = np.mean(task_losses).item()
    coverage = num_covered / num_total
    return avg_task_loss, coverage


def optimize_wrapper_ellipse(
    mlp: MLP,
    quantile_mlp: MLP,
    prob: EllipsoidProblemProtocol,
    loaders: Mapping[str, torch.utils.data.DataLoader],
    q: float,
    L: Tensor,
    device: str = 'cpu',
) -> dict[str, float]:
    mlp.eval().to(device)
    quantile_mlp.eval().to(device)
    result = {'q': q}
    with torch.no_grad():
        for split in ('train', 'test'):
            task_loss, coverage = optimize_ellipse(
                mlp, quantile_mlp, prob=prob, loader=loaders[split], q=q, L=L,
                device=device)
            result |= {
                f'{split}_task_loss': task_loss,
                f'{split}_coverage': coverage,
            }
    return result


def conformal_q_ellipse_johnstone(
    model: nn.Module,
    loaders: Mapping[str, torch.utils.data.DataLoader],
    alpha: float,
    device: str = 'cpu'
) -> tuple[Tensor, Tensor]:
    """
    Args:
        model: outputs point estimate
        ds: dataset
        alpha: desired coverage level in (0, 1)
        device: device to use

    Returns:
        q: scalar score threshold for the desired coverage level
        L: Cholesky factor of covariance matrix for residuals, on device
    """
    model.eval().to(device)

    residuals = []
    for x, y in loaders['train']:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        residuals.append(y - pred)
    residuals = torch.cat(residuals)  # shape [n, y_dim]

    L = torch.linalg.cholesky(torch.cov(residuals.T))  # shape [y_dim, y_dim]

    all_scores = []
    for x, y in loaders['calib']:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        pred = model(x)
        scores = mahalanobis_dist2(pred, scale_tril=L, y=y)
        all_scores.append(scores)
    scores = torch.cat(all_scores)

    q = calc_q(scores, alpha)
    assert q != torch.inf, 'Size of calibration set is too small'
    return q, L


def optimize_ellipse_johnstone(
    model: nn.Module,
    prob: EllipsoidProblemProtocol,
    loader: torch.utils.data.DataLoader,
    q: float,
    L: Tensor,
    device: str = 'cpu',
    show_pbar: bool = False
) -> tuple[float, float]:
    """
    This function should be run inside a `with torch.no_grad()` block.

    Args:
        model: point estimate model, already with .eval() and on device
        prob: robust optimization problem to solve (with box constraints)
        loader: data loader
        q: score threshold for the desired coverage level
        L: Cholesky factor of covariance matrix of residuals
        device: device to use
        show_pbar: whether to show a progress bar

    Returns:
        avg_task_loss: average task loss per training example
        coverage: proportion of covered examples
    """
    L = L.to(device)

    if show_pbar:
        pbar = tqdm(total=len(loader.dataset))

    task_losses = []
    num_total = 0
    num_covered = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        pred = model(x_batch)  # shape [n, y_dim]

        # calculate coverage
        scores = mahalanobis_dist2(pred, L, y_batch)
        covered = (scores <= q)  # shape [n]
        num_covered += int(covered.sum().item())
        num_total += y_batch.shape[0]

        # convert all to numpy
        scale_tril = L.cpu().numpy() * np.sqrt(q)
        loc_batch = pred.cpu().numpy()
        y_batch = y_batch.cpu().numpy()

        for y, loc in zip(y_batch, loc_batch):
            prob.solve(loc, scale_tril)
            task_loss = prob.task_loss_np(y, is_standardized=True)
            task_losses.append(task_loss)
            if show_pbar:
                pbar.update(1)

    avg_task_loss = np.mean(task_losses).item()
    coverage = num_covered / num_total
    return avg_task_loss, coverage


def optimize_wrapper_ellipse_johnstone(
    model: nn.Module,
    prob: EllipsoidProblemProtocol,
    loaders: Mapping[str, torch.utils.data.DataLoader],
    q: float,
    L: Tensor,
    device: str = 'cpu',
) -> dict[str, float]:
    model.eval().to(device)
    result = {'q': q}
    with torch.no_grad():
        for split in ('train', 'test'):
            task_loss, coverage = optimize_ellipse_johnstone(
                model, prob=prob, loader=loaders[split], q=q, L=L, device=device)
            result |= {
                f'{split}_task_loss': task_loss,
                f'{split}_coverage': coverage,
            }
    return result
