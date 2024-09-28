from collections.abc import Mapping
import io
from typing import Any

import numpy as np
import torch.utils.data
from torch import nn, Tensor
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils.conformal import calc_q


class PinballLoss(nn.Module):
    """Pinball (a.k.a. quantile) loss function."""
    def __init__(self, q: float, reduction: str = 'mean'):
        super().__init__()
        self.q = q
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Computes the loss for the given prediction.

        Args:
            input: shape [batch_size, y_dim], predicted quantiles
            target: shape [batch_size, y_dim], true values
        """
        e = target - input
        loss = torch.maximum(self.q * e, (self.q - 1) * e)
        loss = torch.sum(loss, dim=-1)

        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            raise NotImplementedError('Only mean reduction is supported.')


class QuantileRegressor(nn.Module):
    """
    For each input example x of shape [input_dim], the model outputs a vector
    pred of shape [y_dim*2]. The first y_dim elements of pred are the lower
    quantile. The second y_dim elements are a "delta" term that is added to the
    lower quantile to get the upper quantile. To ensure positivity of the
    delta term, it is passed through a softplus.
    """
    def __init__(self, input_dim: int, y_dim: int):
        super().__init__()
        self.y_dim = y_dim
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
            nn.Linear(256, y_dim * 2)
        )

    def forward(self, x: Tensor) -> Tensor:
        pred = self.net(x)  # shape [n, y_dim*2]
        pred_lo = pred[..., :self.y_dim]
        delta = F.softplus(pred[..., self.y_dim:])
        return torch.cat([pred_lo, pred_lo + delta], dim=-1)


def multidim_quantile_score(pred: Tensor, y: Tensor, y_dim: int) -> Tensor:
    """Multidimensional conformal quantile regression (CQR) score function.

    For scalar y, the CQR score function is
        s(x,y) = max{ y - h(x), h(x) - y }
    where h(x) is the predicted quantile.

    For multidimensional y, the CQR score function is
        s(x,y) = max{ max_i(y_i - h(x)_i), max_j(h(x)_j - y_j) }

    Args:
        pred: shape [n, y_dim * 2], model output
        y: shape [n, y_dim], true labels
        y_dim: dimension of each label

    Returns:
        scores: shape [n], quantile scores for each example
    """
    pred_lo = pred[..., :y_dim]
    pred_hi = pred[..., y_dim:]
    scores = torch.maximum(
        torch.max(pred_lo - y, dim=1)[0],
        torch.max(y - pred_hi, dim=1)[0]
    )
    return scores


def conformal_q(
    model: QuantileRegressor,
    loader: torch.utils.data.DataLoader,
    y_dim: int,
    alpha: float,
    device: str = 'cpu'
) -> Tensor:
    """
    Args:
        model: model to evaluate
        loader: data loader
        y_dim: dimension of each label
        alpha: desired coverage level in (0, 1)
        device: device to use

    Returns:
        q: scalar score threshold for the desired coverage level
    """
    model.to(device)
    all_scores = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        scores = multidim_quantile_score(model(x), y, y_dim)
        all_scores.append(scores)

    scores = torch.cat(all_scores)
    q = calc_q(scores, alpha)
    assert q != torch.inf, 'Size of calibration set is too small'
    return q


def run_epoch_quantile_regressor(
    model: QuantileRegressor,
    loader: torch.utils.data.DataLoader,
    y_dim: int,
    alpha: float,
    q: Tensor | float = 0.,
    optimizer: torch.optim.Optimizer | None = None,
    device: str = 'cpu'
) -> tuple[float, float, float]:
    """Runs 1 epoch of training or evaluation with a quantile regression model
    and pinball loss.

    Args:
        model: model to train
        loader: data loader
        y_dim: dimension of each label
        alpha: desired coverage level
        q: optional score threshold for the desired coverage level,
            q=0 is the same as not doing conformal
        optimizer: optimizer to use for training, None for evaluation
        device: either 'cpu' or 'cuda'

    Returns:
        coverage: proportion of targets that fall within predicted quantile range
        coverage_steps: proportion of target steps that fall within predicted quantile range
        avg_loss: average pinball loss per training example
    """
    assert 0 < alpha < 1

    lo_quantile_loss_fn = PinballLoss(alpha/2)
    hi_quantile_loss_fn = PinballLoss(1-alpha/2)

    if optimizer is None:
        model.eval().to(device)
    else:
        model.train().to(device)

    total_loss = 0.
    num_covered = 0.
    num_total = 0
    num_steps_covered = 0.
    num_total_steps = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        pred = model(x)
        pred_lo = pred[..., :y_dim]
        pred_hi = pred[..., y_dim:]

        lo_quantile_loss = lo_quantile_loss_fn(pred_lo, y)
        hi_quantile_loss = hi_quantile_loss_fn(pred_hi, y)
        loss = lo_quantile_loss + hi_quantile_loss
        total_loss += loss.item() * len(x)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            covered = (pred_lo - q <= y) & (y <= pred_hi + q)  # shape [batch, y_dim]

            num_covered += covered.all(dim=1).sum().item()
            num_total += y.shape[0]

            num_steps_covered += covered.sum().item()
            num_total_steps += y.numel()

    coverage = num_covered / num_total
    coverage_steps = num_steps_covered / num_total_steps
    avg_loss = total_loss / num_total
    return coverage, coverage_steps, avg_loss


def train_quantile_regressor(
    loaders: Mapping[str, torch.utils.data.DataLoader],
    input_dim: int,
    y_dim: int,
    alpha: float,
    max_epochs: int,
    lr: float,
    l2reg: float,
    show_pbar: bool = False,
    return_best_model: bool = False,
    device: str = 'cpu'
) -> tuple[QuantileRegressor, dict[str, Any]]:
    """Trains a quantile regression model.

    Uses early-stopping based on quantile regression loss on the calibration set.

    Args:
        loaders: dictionary of data loaders
        input_dim: dimension of each input example x
        y_dim: dimension of each label y
        alpha: risk level in (0, 1), predicted quantiles are α/2 and 1-α/2
        max_epochs: maximum number of epochs to train
        lr: learning rate
        l2reg: L2 regularization strength
        show_pbar: if True, show a progress bar
        return_best_model: if True, return the model with the best validation loss,
            otherwise returns the model from the last training epoch
        device: either 'cpu' or 'cuda'

    Returns:
        model: trained model on CPU, either the model from the last training epoch, or
            the model that achieves result['val_pinball_loss'] on the calibration set,
            see return_best_model
        result: dict of performance metrics
    """
    # Initialize model
    model = QuantileRegressor(input_dim=input_dim, y_dim=y_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2reg)

    result: dict[str, Any] = {
        'train_pinball_losses': [],
        'val_pinball_losses': [],
        'best_epoch': 0,
        'val_pinball_loss': np.inf,  # best loss on val set
        'val_coverage': 0,
        'val_coverage_steps': 0
    }
    steps_since_decrease = 0
    buffer = io.BytesIO()

    for epoch in tqdm(range(max_epochs)) if show_pbar else range(max_epochs):
        # train
        train_cov, train_cov_steps, train_loss = run_epoch_quantile_regressor(
            model, loaders['train'], y_dim=y_dim, alpha=alpha, optimizer=optimizer,
            device=device)
        result['train_pinball_losses'].append(train_loss)

        # calculate loss on calibration set
        with torch.no_grad():
            val_cov, val_cov_steps, val_loss = run_epoch_quantile_regressor(
                model, loaders['calib'], y_dim=y_dim, alpha=alpha, device=device)
            result['val_pinball_losses'].append(val_loss)

        if show_pbar:
            tqdm.write(f'Train cov {train_cov:.3f} cov steps: {train_cov_steps:.3f} loss: {train_loss:.3f}, '
                       f'Calib cov {val_cov:.3f} cov steps: {val_cov_steps:.3f} loss: {val_loss:.3f}')

        steps_since_decrease += 1

        if val_loss < result['val_pinball_loss']:
            result['best_epoch'] = epoch
            result['val_pinball_loss'] = val_loss
            result['val_coverage_no_conformal'] = val_cov
            result['val_coverage_steps_noconformal'] = val_cov_steps
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
